"""Live Genesis frontier mission proof: camera, telemetry, MJPEG, and report."""

import argparse
import io
import json
import os
import random
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


SOUND_SOURCE_WORLD = np.array([2.4, 0.0, 1.0])
ARTIFACTS = (
    "final.json", "frontier_argus_demo.json", "frontier_trajectory.npz",
    "frontier_maps.npz", "occupancy_map.pgm", "sound_heatmap.pgm",
    "frontier_argus_demo.svg", "mission.log",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def gpu_name():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return os.environ.get("ECHOS_GPU_NAME", "unknown")


class LiveState:
    def __init__(self, output_dir, run_uuid, seed, backend, gpu):
        self.output_dir = Path(output_dir)
        self.run_uuid = run_uuid
        self.seed = seed
        self.backend = backend
        self.gpu = gpu
        self.lock = threading.Condition()
        self.frame = None
        self.frame_seq = 0
        self.telemetry = {
            "run_uuid": run_uuid, "seed": seed, "backend": backend,
            "gpu": gpu, "phase": "BOOTSTRAP", "wall_clock_utc": utc_now(),
        }
        self.logs = deque(maxlen=300)
        self.final = None
        self.done = False
        self.log_path = self.output_dir / "mission.log"
        self.log_file = self.log_path.open("w", encoding="utf-8", buffering=1)

    def log(self, message):
        line = f"[{self.run_uuid}] {utc_now()} {message}"
        with self.lock:
            self.logs.append(line)
            self.log_file.write(line + "\n")
            self.log_file.flush()
            self.lock.notify_all()
        print(line, flush=True)

    def publish(self, telemetry, frame):
        with self.lock:
            self.telemetry = dict(telemetry)
            self.telemetry["run_uuid"] = self.run_uuid
            self.telemetry["seed"] = self.seed
            self.telemetry["backend"] = self.backend
            self.telemetry["gpu"] = self.gpu
            self.telemetry["wall_clock_utc"] = utc_now()
            self.frame = frame
            self.frame_seq += 1
            self.telemetry["frame_seq"] = self.frame_seq
            self.lock.notify_all()

    def finish(self, report):
        with self.lock:
            self.final = report
            self.telemetry.update({
                "phase": "COMPLETE",
                "mission_passed": report["mission_passed"],
                "final": report,
            })
            self.done = True
            self.lock.notify_all()
        self.log("COMPLETE mission_passed=%s localization_error_m=%.6f" % (
            report["mission_passed"], report["localization_error_m"]))

    def snapshot(self):
        with self.lock:
            result = dict(self.telemetry)
            result["logs"] = list(self.logs)
            result["final_available"] = self.final is not None
            return result

    def wait_frame(self, previous_seq):
        with self.lock:
            while self.frame_seq <= previous_seq and not self.done:
                self.lock.wait(timeout=1.0)
            return self.frame, self.frame_seq, self.done

    def close(self):
        self.log_file.close()


class LiveRenderer:
    def __init__(self, state, truth, pace_s):
        from PIL import Image, ImageDraw, ImageFont
        from echos_frontier import live_camera, scene

        self.state = state
        self.truth = np.asarray(truth, dtype=float)
        self.pace_s = pace_s
        self.camera = live_camera
        self.scene = scene
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.font = ImageFont.load_default()
        self.last_phase = None
        self.last_frontier = None

    @staticmethod
    def _path(path, limit=700):
        path = np.asarray(path, dtype=float)
        if len(path) <= limit:
            return path
        return path[::max(1, len(path) // limit)]

    def _debug_geometry(self, event, estimate=None):
        self.scene.clear_debug_objects()
        exploration = self._path(event["exploration_path"])
        rtl = self._path(event["rtl_path"])
        if len(exploration) > 1:
            self.scene.draw_debug_trajectory(exploration, radius=0.012,
                                             color=(0.1, 0.9, 0.2, 0.9))
        if len(rtl) > 1:
            self.scene.draw_debug_trajectory(rtl, radius=0.014,
                                             color=(1.0, 0.55, 0.05, 0.95))
        self.scene.draw_debug_sphere((0.0, 0.0, 1.0), radius=0.12,
                                     color=(1.0, 0.7, 0.0, 1.0))
        self.scene.draw_debug_sphere(self.truth, radius=0.11,
                                     color=(0.1, 0.3, 1.0, 1.0))
        frontier = event.get("selected_frontier")
        if frontier is not None:
            target = (frontier["x"], frontier["y"], 1.0)
            self.scene.draw_debug_sphere(target, radius=0.10,
                                         color=(1.0, 0.05, 0.8, 1.0))
            self.scene.draw_debug_line(event["position"], target, radius=0.008,
                                       color=(1.0, 0.05, 0.8, 0.8))
        if estimate is not None:
            self.scene.draw_debug_sphere(estimate, radius=0.10,
                                         color=(1.0, 0.0, 0.0, 1.0))
            self.scene.draw_debug_line(self.truth, estimate, radius=0.006,
                                       color=(1.0, 0.0, 0.0, 0.8))

    def _overlay(self, frame, event, estimate=None, error=None, gates=None):
        image = self.Image.fromarray(frame).convert("RGB")
        draw = self.ImageDraw.Draw(image, "RGBA")
        lines = [
            f"RUN {self.state.run_uuid}  SEED {self.state.seed}",
            f"UTC {utc_now()}",
            f"STEP {event['step']}  SIM {event['sim_time_s']:.2f}s",
            f"BACKEND {self.state.backend}  GPU {self.state.gpu}",
            f"PHASE {event['phase']}",
            "POS (%.3f, %.3f, %.3f)" % tuple(event["position"]),
            "FRONTIER " + ("none" if event["selected_frontier"] is None else
                            "(%.2f, %.2f)" % (event["selected_frontier"]["x"],
                                               event["selected_frontier"]["y"])),
            f"COVERAGE {event['coverage_percent']:.1f}%  COLLISIONS {event['collision_count']}",
            f"CLEARANCE {event['min_clearance_m']:.3f}m  UNKNOWN {event['unknown_traversal']}",
            f"FRONTIERS {event['frontiers_discovered']}  RTL " +
            ("--" if event["rtl_distance_m"] is None else f"{event['rtl_distance_m']:.3f}m"),
            "ARGUS TRUTH (%.2f, %.2f, %.2f)" % tuple(self.truth),
            "ARGUS EST " + ("--" if estimate is None else "(%.3f, %.3f, %.3f)" % tuple(estimate)),
            "ARGUS ERROR " + ("--" if error is None else f"{error:.4f}m"),
        ]
        if gates:
            lines.append("ACCEPTANCE " + ("PASS" if gates["mission_passed"] else "FAIL"))
        width = max(draw.textbbox((0, 0), line, font=self.font)[2] for line in lines) + 18
        draw.rectangle((5, 5, width, 11 + len(lines) * 15), fill=(0, 0, 0, 190))
        for index, line in enumerate(lines):
            color = (120, 255, 140, 255) if line.startswith("ACCEPTANCE PASS") else (255, 255, 255, 255)
            draw.text((12, 9 + index * 15), line, fill=color, font=self.font)
        return np.asarray(image)

    def render(self, event, estimate=None, error=None, gates=None):
        self._debug_geometry(event, estimate)
        rgb = self.camera.render(rgb=True, force_render=True)[0]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if np.issubdtype(rgb.dtype, np.floating) and rgb.max(initial=0) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        frame = self._overlay(rgb, event, estimate, error, gates)
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(frame).save(output, format="JPEG", quality=82)
        self.state.publish({
            "phase": event["phase"], "step": event["step"],
            "sim_time_s": event["sim_time_s"],
            "position_xyz": [float(x) for x in event["position"]],
            "selected_frontier": event["selected_frontier"],
            "coverage_percent": event["coverage_percent"],
            "collision_count": event["collision_count"],
            "min_clearance_m": event["min_clearance_m"],
            "unknown_traversal": event["unknown_traversal"],
            "frontiers_discovered": event["frontiers_discovered"],
            "rtl_distance_m": event["rtl_distance_m"],
            "argus_truth_position": [float(x) for x in self.truth],
            "argus_estimate_position": None if estimate is None else [float(x) for x in estimate],
            "argus_localization_error_m": error,
            "mission_passed": None if gates is None else gates["mission_passed"],
        }, output.getvalue())
        if event["phase"] != self.last_phase:
            self.state.log(f"PHASE {event['phase']} step={event['step']}")
            self.last_phase = event["phase"]
        frontier = event["selected_frontier"]
        if frontier != self.last_frontier and frontier is not None:
            self.state.log(f"FRONTIER x={frontier['x']:.2f} y={frontier['y']:.2f}")
            self.last_frontier = frontier
        time.sleep(self.pace_s * (20 if event["phase"] == "NO_FRONTIER" else 1))


class DashboardHandler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, *_args):
        return

    def _send(self, body, content_type, status=200):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send(DASHBOARD_HTML, "text/html; charset=utf-8")
        elif path == "/status.json":
            self._send(json.dumps(self.state.snapshot()), "application/json")
        elif path == "/final.json":
            if self.state.final is None:
                self._send("final report is not ready\n", "text/plain", 404)
            else:
                self._send(json.dumps(self.state.final), "application/json")
        elif path == "/stream.mjpg":
            self.stream()
        elif path.startswith("/artifacts/"):
            self.artifact(path[len("/artifacts/"):])
        else:
            self._send("not found\n", "text/plain", 404)

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        previous = 0
        try:
            while True:
                frame, sequence, done = self.state.wait_frame(previous)
                if frame is not None and sequence > previous:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame + b"\r\n")
                    self.wfile.flush()
                    previous = sequence
                if done and sequence == previous:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    def artifact(self, relative):
        root = self.state.output_dir.resolve()
        candidate = (root / relative).resolve()
        if root not in candidate.parents and candidate != root:
            self._send("forbidden\n", "text/plain", 403)
            return
        if candidate == root:
            links = "".join(f'<li><a href="/artifacts/{name}">{name}</a></li>' for name in ARTIFACTS)
            self._send(f"<html><body><ul>{links}</ul></body></html>", "text/html; charset=utf-8")
            return
        if not candidate.is_file():
            self._send("not found\n", "text/plain", 404)
            return
        content_types = {".json": "application/json", ".svg": "image/svg+xml", ".pgm": "image/x-portable-graymap", ".npz": "application/octet-stream", ".log": "text/plain"}
        self._send(candidate.read_bytes(), content_types.get(candidate.suffix, "application/octet-stream"))


DASHBOARD_HTML = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Genesis Frontier Live Proof</title>
<style>body{background:#101318;color:#e9edf1;font:14px monospace;margin:16px}main{display:grid;grid-template-columns:minmax(480px,2fr) minmax(300px,1fr);gap:16px}img{width:100%;background:#000;border:1px solid #39414d}pre{white-space:pre-wrap;background:#171c24;padding:12px;overflow:auto;max-height:360px}h1{font-size:20px}.pass{color:#7dff98}.fail{color:#ff7777}@media(max-width:900px){main{display:block}}</style></head>
<body><h1>Genesis Frontier Live Proof</h1><main><section><img src="/stream.mjpg"><h2>Mission Log</h2><pre id="logs">waiting for mission...</pre></section>
<section><h2>Telemetry</h2><pre id="telemetry">connecting...</pre><h2>Final Gates</h2><pre id="gates">pending...</pre><h2>Artifacts</h2><ul id="artifacts"></ul></section></main>
<script>
const names=['final.json','frontier_argus_demo.json','frontier_trajectory.npz','frontier_maps.npz','occupancy_map.pgm','sound_heatmap.pgm','frontier_argus_demo.svg','mission.log'];
document.getElementById('artifacts').innerHTML=names.map(n=>`<li><a href="/artifacts/${n}">${n}</a></li>`).join('');
async function poll(){try{const s=await (await fetch('/status.json',{cache:'no-store'})).json();
document.getElementById('telemetry').textContent=JSON.stringify(s,null,2); document.getElementById('logs').textContent=(s.logs||[]).join('\n'); const l=document.getElementById('logs'); l.scrollTop=l.scrollHeight;
if(s.final_available){const f=await (await fetch('/final.json',{cache:'no-store'})).json(); document.getElementById('gates').textContent=JSON.stringify(f.acceptance_gates||f,null,2); document.title=(f.mission_passed?'PASS':'FAIL')+' - Genesis Frontier';}}
catch(e){document.getElementById('telemetry').textContent='server unavailable: '+e;} } setInterval(poll,500); poll();
</script></body></html>"""


def enable_tailscale(port):
    candidates = ["tailscale", "/mnt/c/Program Files/Tailscale/tailscale.exe"]
    command = next((path for path in candidates if Path(path).exists() or path == "tailscale"), None)
    if command is None:
        return None
    result = subprocess.run([command, "serve", "--bg", str(port)], capture_output=True,
                            text=True, check=False)
    output = result.stdout + result.stderr
    match = re.search(r"https://[^\s]+", output)
    if match:
        return match.group(0).rstrip("/\r\n")
    status = subprocess.run([command, "serve", "status", "--json"], capture_output=True,
                            text=True, check=False)
    try:
        payload = json.loads(status.stdout)
        text = json.dumps(payload)
        match = re.search(r"https://[^\" ]+", text)
        if match:
            return match.group(0).rstrip("/")
    except json.JSONDecodeError:
        pass
    return None


def write_artifacts(output_dir, mission, estimate, truth, error, run_uuid, seed, backend, gpu):
    from frontier_argus_demo import _write_pgm, _write_svg

    output_dir = Path(output_dir)
    mapper = mission["mapper"]
    occupancy = mapper.get_map()
    heatmap = np.zeros_like(occupancy, dtype=float)
    ix, iy = mapper.w2g(estimate[0], estimate[1])
    if mapper.in_b(ix, iy):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if mapper.in_b(ix + dx, iy + dy):
                    heatmap[iy + dy, ix + dx] = np.exp(-0.5 * (dx * dx + dy * dy))
    np.savez_compressed(output_dir / "frontier_maps.npz", occupancy=occupancy,
                        views=mapper.views, sound_heatmap=heatmap,
                        trajectory=mission["trajectory"], rtl_trajectory=mission["rtl_trajectory"],
                        source_truth=truth, source_estimate=estimate)
    _write_pgm(output_dir / "occupancy_map.pgm", occupancy)
    _write_pgm(output_dir / "sound_heatmap.pgm", heatmap)
    _write_svg(output_dir / "frontier_argus_demo.svg", mapper, occupancy,
               mission["trajectory"], mission["rtl_trajectory"], truth, estimate)
    gates = {
        "explored_ge_95_percent": mission["coverage_percent"] >= 95.0,
        "no_collision": not mission["collision"],
        "clearance_gt_0_20_m": mission["min_clearance"] > 0.20,
        "frontiers_found": len(mission["frontiers"]) > 0,
        "rtl_within_5_cm": mission["rtl_dist"] < 0.05,
        "both_legs_discovered": mission["passed"],
        "zero_unknown_traversal": mission["unknown_traversal"] == 0,
        "no_frontier_oscillation": not mission["oscillation"],
        "terminated_by_no_frontier": mission["terminated_no_frontier"],
    }
    report = {
        "run_uuid": run_uuid, "seed": seed, "backend": backend, "gpu": gpu,
        "mission_passed": bool(mission["passed"] and error < 0.1),
        "terminated_no_frontier": bool(mission["terminated_no_frontier"]),
        "coverage_percent": float(mission["coverage_percent"]),
        "collision": bool(mission["collision"]),
        "collision_count": int(mission["collision_count"]),
        "min_clearance_m": mission["min_clearance"],
        "unknown_traversal": mission["unknown_traversal"],
        "frontier_oscillation": mission["oscillation"],
        "frontier_centers": [[float(f["cx"]), float(f["cy"])] for f in mission["frontiers"]],
        "rtl_distance_m": float(mission["rtl_dist"]),
        "source_truth_world_m": truth.tolist(),
        "source_estimate_world_m": estimate.tolist(),
        "localization_error_m": error,
        "trajectory_path": str(output_dir / "frontier_trajectory.npz"),
        "acceptance_gates": gates,
    }
    (output_dir / "frontier_argus_demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "final.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/workspace/frontier_live_demo")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--pace", type=float, default=0.08, help="seconds per streamed frame")
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--no-tailscale-serve", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("--host must remain 127.0.0.1")
    if args.port < 1 or args.render_every < 1 or args.pace < 0:
        raise SystemExit("port, render interval, and pace must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("ECHOS_GENESIS_BACKEND", "gpu")
    os.environ["ECHOS_SHOW_VIEWER"] = "1" if args.viewer else "0"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_uuid = str(uuid.uuid4())
    backend = os.environ["ECHOS_GENESIS_BACKEND"]
    gpu = gpu_name()

    from echos_frontier import run_frontier_exploration
    from replay_argus import replay

    state = LiveState(output_dir, run_uuid, args.seed, backend, gpu)
    renderer = LiveRenderer(state, SOUND_SOURCE_WORLD, args.pace)
    DashboardHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.log(f"START seed={args.seed} backend={backend} gpu={gpu} viewer={args.viewer}")
    print(f"Local dashboard: http://127.0.0.1:{args.port}/", flush=True)
    if not args.no_tailscale_serve:
        tailnet_url = enable_tailscale(args.port)
        if tailnet_url:
            print(f"Tailnet URL: {tailnet_url}/", flush=True)
            state.log(f"TAILNET_URL {tailnet_url}/")
        else:
            print("Tailnet URL: unavailable; run tailscale serve %d" % args.port, flush=True)

    last_phase = [None]

    def on_event(event):
        if event["phase"] != last_phase[0]:
            state.log(f"EVENT phase={event['phase']} step={event['step']}")
            last_phase[0] = event["phase"]
        renderer.render(event)

    trajectory_path = output_dir / "frontier_trajectory.npz"
    try:
        mission = run_frontier_exploration(record_path=trajectory_path,
                                            live_callback=on_event,
                                            live_interval=args.render_every)
        state.log("REPLAY ARGUS starting")
        estimate, truth = replay(str(trajectory_path), target=SOUND_SOURCE_WORLD)
        error = float(np.linalg.norm(estimate - truth))
        report = write_artifacts(output_dir, mission, estimate, truth, error,
                                 run_uuid, args.seed, backend, gpu)
        final_event = {
            "phase": "COMPLETE", "step": mission["term_step"] + 21 + len(mission["rtl_trajectory"]),
            "sim_time_s": float(renderer.state.telemetry.get("sim_time_s", 0.0)),
            "position": mission["rtl_trajectory"][-1] if len(mission["rtl_trajectory"]) else np.array([0.0, 0.0, 1.0]),
            "coverage_percent": mission["coverage_percent"],
            "collision_count": mission["collision_count"],
            "min_clearance_m": mission["min_clearance"],
            "unknown_traversal": mission["unknown_traversal"],
            "frontiers_discovered": len(mission["frontiers"]),
            "selected_frontier": None, "rtl_distance_m": mission["rtl_dist"],
            "exploration_path": mission["trajectory"], "rtl_path": mission["rtl_trajectory"],
        }
        renderer.render(final_event, estimate=estimate, error=error, gates=report)
        state.finish(report)
        state.log(f"FINAL_JSON {output_dir / 'final.json'}")
        if not report["mission_passed"]:
            raise SystemExit(1)
    except Exception as exc:
        state.log(f"FAIL {type(exc).__name__}: {exc}")
        raise
    finally:
        server.shutdown()
        state.close()


if __name__ == "__main__":
    main()
