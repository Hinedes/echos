"""Live Genesis frontier mission proof: camera, telemetry, MJPEG, and report."""

import argparse
import io
import json
import os
import random
import re
import shutil
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
    "contact_failure.json", "contact_failure_frame.jpg",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_text(value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    return json.dumps(clean(value), allow_nan=False)


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
        self.telemetry = {"run_uuid": run_uuid, "seed": seed, "backend": backend,
                          "gpu": gpu, "phase": "IDLE", "step": 0,
                          "sim_time_s": 0.0, "mission_started": False,
                          "paused": False, "control_state": "WAITING_FOR_START",
                          "position_xyz": [0.0, 0.0, 1.0], "coverage_percent": 0.0,
                          "physical_clearance_m": None, "sensor_range_m": None,
                          "active_contact_count": 0, "contact_count": 0,
                          "last_contact": None, "safety_stop_active": False,
                          "rtl_distance_m": None,
                          "selected_frontier": None, "exploration_path": [], "rtl_path": [],
                          "wall_clock_utc": utc_now()}
        self.logs = deque(maxlen=300)
        self.final = None
        self.done = False
        self.started = False
        self.paused = False
        self.abort_requested = False
        self.camera_name = "overhead"
        self.start_callback = None
        self.shutdown_event = threading.Event()
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
            self.telemetry["mission_started"] = self.started
            self.telemetry["paused"] = self.paused
            self.telemetry["control_state"] = "PAUSED" if self.paused else self.telemetry.get("phase", "IDLE")
            self.telemetry["camera"] = self.camera_name
            self.frame = frame
            self.frame_seq += 1
            self.telemetry["frame_seq"] = self.frame_seq
            self.lock.notify_all()

    def finish(self, report):
        with self.lock:
            self.final = report
            phase = "CONTACT_FAIL" if report.get("contact_failure") else "ABORTED" if report.get("aborted") else "COMPLETE"
            self.telemetry.update({
                "phase": phase,
                "mission_passed": report["mission_passed"],
                "control_state": phase,
                "final": report,
            })
            self.done = True
            self.lock.notify_all()
        message = "CONTACT_FAIL" if report.get("contact_failure") else "ABORTED" if report.get("aborted") else "COMPLETE"
        if "localization_error_m" in report:
            message += " mission_passed=%s localization_error_m=%.6f" % (
                report["mission_passed"], report["localization_error_m"])
        self.log(message)

    def snapshot(self):
        with self.lock:
            result = dict(self.telemetry)
            result["wall_clock_utc"] = utc_now()
            result["paused"] = self.paused
            result["control_state"] = "PAUSED" if self.paused else result.get("control_state", "IDLE")
            result["camera"] = self.camera_name
            result["logs"] = list(self.logs)
            result["final_available"] = self.final is not None
            return result

    def wait_frame(self, previous_seq):
        with self.lock:
            while self.frame_seq <= previous_seq and not self.shutdown_event.is_set():
                self.lock.wait(timeout=1.0)
            return self.frame, self.frame_seq, self.shutdown_event.is_set()

    def start(self):
        with self.lock:
            if self.started:
                return False
            self.started = True
            self.telemetry.update({"phase": "BOOTSTRAP", "mission_started": True,
                                   "control_state": "BOOTSTRAP"})
            callback = self.start_callback
        self.log("START_REQUEST accepted")
        callback()
        return True

    def set_paused(self, paused):
        with self.lock:
            if not self.started or self.done:
                return False
            if self.paused == paused:
                return True
            self.paused = paused
            self.lock.notify_all()
        self.log("PAUSE_REQUEST" if paused else "RESUME_REQUEST")
        return True

    def abort(self):
        with self.lock:
            if not self.started or self.done:
                return False
            self.abort_requested = True
            self.paused = False
            self.lock.notify_all()
        self.log("ABORT_REQUEST accepted")
        return True

    def reset(self):
        with self.lock:
            if self.started and not self.done:
                self.abort_requested = True
                self.paused = False
                self.lock.notify_all()
                deadline = time.monotonic() + 5.0
                while not self.done and time.monotonic() < deadline:
                    self.lock.wait(timeout=0.1)
                if not self.done:
                    return False
            self.run_uuid = str(uuid.uuid4())
            self.frame = None
            self.frame_seq = 0
            self.telemetry = {
                "run_uuid": self.run_uuid, "seed": self.seed,
                "backend": self.backend, "gpu": self.gpu, "phase": "IDLE",
                "step": 0, "sim_time_s": 0.0, "mission_started": False,
                "paused": False, "control_state": "WAITING_FOR_START",
                "position_xyz": [0.0, 0.0, 1.0], "coverage_percent": 0.0,
                "physical_clearance_m": None, "sensor_range_m": None,
                "active_contact_count": 0, "contact_count": 0,
                "last_contact": None, "safety_stop_active": False,
                "rtl_distance_m": None,
                "selected_frontier": None, "exploration_path": [], "rtl_path": [],
                "wall_clock_utc": utc_now(), "camera": self.camera_name,
            }
            self.logs.clear()
            self.final = None
            self.done = False
            self.started = False
            self.paused = False
            self.abort_requested = False
            self.lock.notify_all()
        for name in ("final.json", "frontier_argus_demo.json"):
            try:
                (self.output_dir / name).unlink()
            except FileNotFoundError:
                pass
        self.log("RESET_REQUEST accepted")
        return True

    def step_guard(self):
        with self.lock:
            while self.paused and not self.abort_requested:
                self.lock.wait(timeout=0.5)
            return not self.abort_requested

    def set_camera(self, name):
        with self.lock:
            self.camera_name = name
            self.telemetry["camera"] = name

    def close(self):
        self.shutdown_event.set()
        with self.lock:
            self.lock.notify_all()
        self.log_file.close()


class LiveRenderer:
    def __init__(self, state, truth, pace_s):
        from PIL import Image, ImageDraw, ImageFont
        from echos_frontier import (
            body, live_camera, physical_body_clearance, physical_contacts,
            reset_body_state, rs, scene,
        )

        self.state = state
        self.truth = np.asarray(truth, dtype=float)
        self.pace_s = pace_s
        self.camera = live_camera
        self.scene = scene
        self.body = body
        self.rs = rs
        self.reset_body_state = reset_body_state
        self.physical_body_clearance = physical_body_clearance
        self.physical_contacts = physical_contacts
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.font = ImageFont.load_default()
        self.last_phase = None
        self.last_frontier = None
        self.last_event = None
        self.last_estimate = None
        self.last_error = None
        self.last_gates = None
        self.heartbeat = False
        self.render_lock = threading.Lock()
        self.camera_poses = (
            ((3.0, -7.5, 8.5), (2.7, 1.0, 0.0)),
            ((7.5, 5.5, 6.0), (2.5, 1.0, 0.6)),
        )

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
        self.scene.draw_debug_sphere(event["position"], radius=0.15,
                                     color=(0.1, 1.0, 0.95, 1.0))
        self.scene.draw_debug_arrow(event["position"], vec=(0.0, 0.0, 0.55),
                                    radius=0.018, color=(0.1, 1.0, 0.95, 1.0))
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
            f"FRAME {self.state.frame_seq + 1}  HEARTBEAT {'|' if self.heartbeat else '/'}",
            f"UTC {utc_now()}",
            f"STEP {event['step']}  SIM {event['sim_time_s']:.2f}s",
            f"BACKEND {self.state.backend}  GPU {self.state.gpu}",
            f"PHASE {event['phase']}",
            f"CONTROL {'PAUSED' if self.state.paused else self.state.telemetry.get('control_state', 'RUNNING')}",
            "POS (%.3f, %.3f, %.3f)" % tuple(event["position"]),
            "FRONTIER " + ("none" if event["selected_frontier"] is None else
                            "(%.2f, %.2f)" % (event["selected_frontier"]["x"],
                                               event["selected_frontier"]["y"])),
            f"COVERAGE {event['coverage_percent']:.1f}%  COLLISIONS {event['collision_count']}",
            "PHYS " + ("--" if event["physical_clearance_m"] is None or not np.isfinite(event["physical_clearance_m"]) else f"{event['physical_clearance_m']:.3f}m") +
            "  SENSOR " + ("--" if event["sensor_range_m"] is None or not np.isfinite(event["sensor_range_m"]) else f"{event['sensor_range_m']:.3f}m"),
            f"CONTACTS {event['active_contact_count']} / TOTAL {event['contact_count']}  UNKNOWN {event['unknown_traversal']}",
            f"SAFETY_STOP {event['safety_stop_active']}",
            f"FRONTIERS {event['frontiers_discovered']}  RTL " +
            ("--" if event["rtl_distance_m"] is None else f"{event['rtl_distance_m']:.3f}m"),
            "ARGUS TRUTH (%.2f, %.2f, %.2f)" % tuple(self.truth),
            "ARGUS EST " + ("--" if estimate is None else "(%.3f, %.3f, %.3f)" % tuple(estimate)),
            "ARGUS ERROR " + ("--" if error is None else f"{error:.4f}m"),
        ]
        if gates and event["phase"] == "COMPLETE":
            lines.append("ACCEPTANCE " + ("PASS" if gates["mission_passed"] else "FAIL"))
        width = max(draw.textbbox((0, 0), line, font=self.font)[2] for line in lines) + 18
        draw.rectangle((5, 5, width, 11 + len(lines) * 15), fill=(0, 0, 0, 190))
        for index, line in enumerate(lines):
            color = (120, 255, 140, 255) if line.startswith("ACCEPTANCE PASS") else (255, 255, 255, 255)
            draw.text((12, 9 + index * 15), line, fill=color, font=self.font)
        return np.asarray(image)

    def render(self, event, estimate=None, error=None, gates=None):
        with self.render_lock:
            self._render(event, estimate, error, gates)

    def _render(self, event, estimate=None, error=None, gates=None):
        self.last_event = dict(event)
        self.last_estimate = None if estimate is None else np.asarray(estimate, dtype=float).copy()
        self.last_error = error
        self.last_gates = gates
        self.heartbeat = not self.heartbeat
        if self.state.camera_name == "alternate":
            pos = np.asarray(event["position"], dtype=float)
            self.camera.set_pose(pos=pos + np.array([2.2, -2.8, 2.0]),
                                 lookat=pos, up=(0.0, 0.0, 1.0))
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
        jpeg = output.getvalue()
        self.state.publish({
            "phase": event["phase"], "step": event["step"],
            "sim_time_s": event["sim_time_s"],
            "position_xyz": [float(x) for x in event["position"]],
            "selected_frontier": event["selected_frontier"],
            "coverage_percent": event["coverage_percent"],
            "collision_count": event["collision_count"],
            "physical_clearance_m": event["physical_clearance_m"],
            "sensor_range_m": event["sensor_range_m"],
            "active_contact_count": event["active_contact_count"],
            "contact_count": event["contact_count"],
            "last_contact": event["last_contact"],
            "contacts": event.get("contacts", []),
            "safety_stop_active": event["safety_stop_active"],
            "unknown_traversal": event["unknown_traversal"],
            "frontiers_discovered": event["frontiers_discovered"],
            "rtl_distance_m": event["rtl_distance_m"],
            "argus_truth_position": [float(x) for x in self.truth],
            "argus_estimate_position": None if estimate is None else [float(x) for x in estimate],
            "argus_localization_error_m": error,
            "mission_passed": None if gates is None else gates["mission_passed"],
            "exploration_path": [[float(p[0]), float(p[1])] for p in self._path(event["exploration_path"], 250)],
            "rtl_path": [[float(p[0]), float(p[1])] for p in self._path(event["rtl_path"], 250)],
        }, jpeg)
        if event["phase"] == "CONTACT_FAIL":
            (self.state.output_dir / "contact_failure_frame.jpg").write_bytes(jpeg)
            (self.state.output_dir / "contact_failure.json").write_text(
                json.dumps({"run_uuid": self.state.run_uuid, "step": event["step"],
                            "sim_time_s": event["sim_time_s"], "contacts": event.get("contacts", []),
                            "physical_clearance_m": event["physical_clearance_m"]}, indent=2),
                encoding="utf-8")
        if event["phase"] != self.last_phase:
            self.state.log(f"PHASE {event['phase']} step={event['step']}")
            self.last_phase = event["phase"]
        frontier = event["selected_frontier"]
        if frontier != self.last_frontier and frontier is not None:
            self.state.log(f"FRONTIER x={frontier['x']:.2f} y={frontier['y']:.2f}")
            self.last_frontier = frontier
        time.sleep(self.pace_s * (20 if event["phase"] == "NO_FRONTIER" else 1))

    def render_heartbeat(self):
        if self.last_event is None:
            return
        self.render(self.last_event, self.last_estimate, self.last_error, self.last_gates)

    def toggle_camera(self):
        with self.render_lock:
            self.state.set_camera("alternate" if self.state.camera_name == "overhead" else "overhead")
            if self.state.camera_name == "overhead":
                pos, lookat = self.camera_poses[0]
                self.camera.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))
            if self.last_event is not None:
                self._render(self.last_event, self.last_estimate, self.last_error, self.last_gates)

    def reset_view(self):
        with self.render_lock:
            self.reset_body_state(self.body, self.rs, pos=(0.0, 0.0, 1.0))
            self.scene.step()
            contacts = self.physical_contacts()
            self.reset_body_state(self.body, self.rs, pos=(0.0, 0.0, 1.0))
            clearance, _ = self.physical_body_clearance(
                self.body.get_pos().cpu().numpy(), self.body.get_quat().cpu().numpy())
            self.state.set_camera("overhead")
            self.last_event = None
            self.last_estimate = None
            self.last_error = None
            self.last_gates = None
            self._render({
                "phase": "IDLE", "step": 0, "sim_time_s": 0.0,
                "position": np.array([0.0, 0.0, 1.0]), "coverage_percent": 0.0,
                "collision_count": 0, "physical_clearance_m": clearance,
                "sensor_range_m": None, "active_contact_count": 0,
                "contact_count": len(contacts),
                "last_contact": contacts[-1] if contacts else None,
                "contacts": contacts, "safety_stop_active": False,
                "unknown_traversal": 0, "frontiers_discovered": 0,
                "selected_frontier": None, "rtl_distance_m": None,
                "exploration_path": np.empty((0, 3)), "rtl_path": np.empty((0, 3)),
            })


class DashboardHandler(BaseHTTPRequestHandler):
    state = None
    renderer = None

    def log_message(self, *_args):
        return

    def _send(self, body, content_type, status=200):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send(DASHBOARD_HTML, "text/html; charset=utf-8")
        elif path == "/status.json":
            self._send(json_text(self.state.snapshot()), "application/json")
        elif path == "/final.json":
            if self.state.final is None:
                self._send("final report is not ready\n", "text/plain", 404)
            else:
                self._send(json_text(self.state.final), "application/json")
        elif path == "/stream.mjpg":
            self.stream()
        elif path.startswith("/artifacts/"):
            self.artifact(path[len("/artifacts/"):])
        else:
            self._send("not found\n", "text/plain", 404)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        actions = {
            "/control/start": lambda: self.state.start(),
            "/control/pause": lambda: self.state.set_paused(True),
            "/control/resume": lambda: self.state.set_paused(False),
            "/control/abort": lambda: self.state.abort(),
            "/control/camera": lambda: (self.renderer.toggle_camera() or True),
            "/control/reset": lambda: self._reset(),
        }
        action = actions.get(path)
        if action is None:
            self._send(json_text({"ok": False, "error": "not found"}), "application/json", 404)
            return
        try:
            changed = action()
            self._send(json_text({"ok": True, "changed": bool(changed)}), "application/json")
        except Exception as exc:
            self._send(json_text({"ok": False, "error": str(exc)}), "application/json", 500)

    def _reset(self):
        changed = self.state.reset()
        if changed:
            self.renderer.reset_view()
        return changed

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
<style>body{background:#0d1117;color:#e9edf1;font:14px monospace;margin:16px}h1{font-size:21px;margin:0 0 8px}.control{padding:8px 0;border-bottom:1px solid #303844}button{background:#2b6cff;color:white;border:0;border-radius:4px;padding:9px 13px;margin:3px;font:inherit;cursor:pointer}button:disabled{background:#444;cursor:not-allowed}.alarm{background:#7f1010;color:#fff;padding:12px;margin-top:10px;font-weight:bold}.layout{display:grid;grid-template-columns:minmax(620px,2fr) minmax(330px,1fr);gap:14px;margin-top:14px}.viewport,.map{background:#05070a;border:1px solid #39414d;padding:5px}.viewport img{display:block;width:100%;min-height:360px;object-fit:cover}.map canvas{display:block;width:100%;height:320px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px}.card{background:#171c24;border-radius:4px;padding:10px}.label{color:#8b98aa;font-size:11px}.value{font-size:18px;margin-top:4px;color:#fff}.pass{color:#7dff98}.fail{color:#ff7777}pre{white-space:pre-wrap;background:#171c24;padding:12px;overflow:auto;max-height:300px}details{margin-top:12px}a{color:#70a7ff}@media(max-width:1000px){.layout{display:block}.map{margin-top:14px}}</style></head>
<body><h1>Genesis Frontier Live Proof</h1>
<div class="control"><button id="start">Start mission</button><button id="pause">Pause</button><button id="resume">Resume</button><button id="abort">Abort</button><button id="reset">Reset</button><button id="camera">Move camera</button><span id="action">WAITING FOR START</span></div><div id="alarm" class="alarm" hidden></div>
<div class="layout"><section><div class="viewport"><img id="stream" src="/stream.mjpg" alt="live Genesis camera"></div><div class="map"><canvas id="map" width="900" height="500"></canvas></div></section>
<aside><div class="cards"><div class="card"><div class="label">PHASE</div><div class="value" id="phase">IDLE</div></div><div class="card"><div class="label">SIM STEP</div><div class="value" id="step">0</div></div><div class="card"><div class="label">POSITION XYZ</div><div class="value" id="position">-</div></div><div class="card"><div class="label">FRAME / HEARTBEAT</div><div class="value" id="frame">-</div></div><div class="card"><div class="label">COVERAGE</div><div class="value" id="coverage">-</div></div><div class="card"><div class="label">FRONTIER</div><div class="value" id="frontier">-</div></div><div class="card"><div class="label">CLEARANCE</div><div class="value" id="clearance">-</div></div><div class="card"><div class="label">RTL / ARGUS ERROR</div><div class="value" id="results">-</div></div></div><h2>Final Gates</h2><pre id="gates">pending...</pre><h2>Artifacts</h2><ul id="artifacts"></ul></aside></div>
<details><summary>Raw telemetry and mission log</summary><pre id="telemetry">connecting...</pre><pre id="logs">waiting for mission...</pre></details>
<script>
(function () {
  var names = ['final.json','frontier_argus_demo.json','frontier_trajectory.npz','frontier_maps.npz','occupancy_map.pgm','sound_heatmap.pgm','frontier_argus_demo.svg','mission.log'];
  var ids = {telemetry: document.getElementById('telemetry'), logs: document.getElementById('logs'), gates: document.getElementById('gates'), action: document.getElementById('action'), alarm: document.getElementById('alarm'), phase: document.getElementById('phase'), position: document.getElementById('position'), step: document.getElementById('step'), frame: document.getElementById('frame'), coverage: document.getElementById('coverage'), frontier: document.getElementById('frontier'), clearance: document.getElementById('clearance'), results: document.getElementById('results')};
  var buttons = {start: document.getElementById('start'), pause: document.getElementById('pause'), resume: document.getElementById('resume'), abort: document.getElementById('abort'), reset: document.getElementById('reset'), camera: document.getElementById('camera')};
  var canvas = document.getElementById('map');
  var ctx = canvas.getContext('2d');
  document.getElementById('artifacts').innerHTML = names.map(function (n) { return '<li><a href="/artifacts/' + n + '">' + n + '</a></li>'; }).join('');
  function control(path) {
    fetch(path, {method: 'POST', cache: 'no-store'}).then(function () { poll(); }).catch(function (e) { ids.action.textContent = String(e); });
  }
  buttons.start.onclick = function () { control('/control/start'); };
  buttons.pause.onclick = function () { control('/control/pause'); };
  buttons.resume.onclick = function () { control('/control/resume'); };
  buttons.abort.onclick = function () { control('/control/abort'); };
  buttons.reset.onclick = function () { control('/control/reset'); };
  buttons.camera.onclick = function () { control('/control/camera'); };
  function mapPoint(p) { return [30 + (p[0] + 2) * (canvas.width - 60) / 9, canvas.height - 25 - (p[1] + 2) * (canvas.height - 50) / 7]; }
  function drawMap(s) {
    ctx.fillStyle = '#101820'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#263746'; ctx.lineWidth = 1;
    for (var gx = -2; gx <= 7; gx += 1) { var a = mapPoint([gx, -2]); var b = mapPoint([gx, 5]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    for (var gy = -2; gy <= 5; gy += 1) { var c = mapPoint([-2, gy]); var d = mapPoint([7, gy]); ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke(); }
    function path(points, color, width) { if (!points || points.length < 2) return; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); points.forEach(function (p, i) { var q = mapPoint(p); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); }); ctx.stroke(); }
    path(s.exploration_path, '#2cff72', 4); path(s.rtl_path, '#ff9f1c', 4);
    function dot(point, color, radius) { var q = mapPoint(point); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(q[0], q[1], radius, 0, Math.PI * 2); ctx.fill(); }
    dot([0, 0], '#ffc400', 8); dot([2.4, 0], '#4a8cff', 8); dot([s.position_xyz[0], s.position_xyz[1]], '#16f0e5', 10);
    if (s.selected_frontier) dot([s.selected_frontier.x, s.selected_frontier.y], '#ff35d0', 9);
    ctx.fillStyle = '#e9edf1'; ctx.font = '16px monospace'; ctx.fillText('LIVE FLIGHT MAP  green=explore  orange=RTL  cyan=drone', 16, 24);
  }
  function poll() {
    fetch('/status.json?ts=' + Date.now(), {cache: 'no-store'}).then(function (response) { return response.json(); }).then(function (s) {
      ids.telemetry.textContent = JSON.stringify(s, null, 2);
      ids.logs.textContent = (s.logs || []).join(String.fromCharCode(10));
      ids.logs.scrollTop = ids.logs.scrollHeight;
      ids.action.textContent = s.control_state || s.phase;
      ids.alarm.hidden = !(s.phase === 'CONTACT_FAIL' || Number(s.contact_count || 0) > 0 || (s.physical_clearance_m != null && Number(s.physical_clearance_m) <= 0.2));
      ids.alarm.textContent = ids.alarm.hidden ? '' : 'PHYSICAL SAFETY FAILURE: CONTACT / CLEARANCE GATE FAILED';
      ids.phase.textContent = s.phase + (s.paused ? ' / PAUSED' : '');
      ids.step.textContent = String(s.step);
      ids.position.textContent = s.position_xyz.map(function (v) { return Number(v).toFixed(2); }).join(', ');
      ids.frame.textContent = String(s.frame_seq) + ' / ' + (s.frame_seq % 2 ? '|' : '/');
      ids.coverage.textContent = Number(s.coverage_percent).toFixed(1) + '%';
      ids.frontier.textContent = s.selected_frontier ? Number(s.selected_frontier.x).toFixed(2) + ', ' + Number(s.selected_frontier.y).toFixed(2) : 'none';
      ids.clearance.textContent = (s.physical_clearance_m == null ? '-' : Number(s.physical_clearance_m).toFixed(3) + ' m') + ' / sensor ' + (s.sensor_range_m == null ? '-' : Number(s.sensor_range_m).toFixed(3) + ' m');
      ids.results.textContent = (s.rtl_distance_m == null ? '-' : Number(s.rtl_distance_m).toFixed(3) + ' m') + ' / ' + (s.argus_localization_error_m == null ? '-' : Number(s.argus_localization_error_m).toFixed(3) + ' m');
      drawMap(s);
      buttons.start.disabled = s.mission_started || s.final_available;
      buttons.pause.disabled = !s.mission_started || s.paused || s.final_available;
      buttons.resume.disabled = !s.mission_started || !s.paused || s.final_available;
      buttons.abort.disabled = !s.mission_started || s.final_available;
      buttons.reset.disabled = s.mission_started && !s.final_available;
      if (s.final_available) {
        fetch('/final.json?ts=' + Date.now(), {cache: 'no-store'}).then(function (response) { return response.json(); }).then(function (f) {
          ids.gates.textContent = JSON.stringify(f.acceptance_gates || f, null, 2);
          document.title = (f.mission_passed ? 'PASS' : 'FAIL') + ' - Genesis Frontier';
        });
      }
    }).catch(function (e) { ids.telemetry.textContent = 'status error: ' + String(e); });
  }
  setInterval(poll, 500);
  poll();
}());
</script></body></html>"""


def enable_tailscale(port):
    candidates = ["tailscale", "/mnt/c/Program Files/Tailscale/tailscale.exe"]
    command = next((path for path in candidates if (
        shutil.which(path) if path == "tailscale" else Path(path).exists()
    )), None)
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
        "explored_ge_95_percent": bool(mission["coverage_percent"] >= 95.0),
        "no_collision": bool(not mission["collision"]),
        "clearance_gt_0_20_m": bool(mission["min_clearance"] > 0.20),
        "frontiers_found": bool(len(mission["frontiers"]) > 0),
        "rtl_within_5_cm": bool(mission["rtl_dist"] < 0.05),
        "both_legs_discovered": bool(mission["both_legs"]),
        "zero_unknown_traversal": bool(mission["unknown_traversal"] == 0),
        "no_frontier_oscillation": bool(not mission["oscillation"]),
        "terminated_by_no_frontier": bool(mission["terminated_no_frontier"]),
    }
    report = {
        "run_uuid": run_uuid, "seed": seed, "backend": backend, "gpu": gpu,
        "mission_passed": bool(mission["passed"] and error < 0.1),
        "terminated_no_frontier": bool(mission["terminated_no_frontier"]),
        "coverage_percent": float(mission["coverage_percent"]),
        "collision": bool(mission["collision"]),
        "collision_count": int(mission["collision_count"]),
        "physical_clearance_m": mission["physical_clearance_m"],
        "sensor_range_m": mission["sensor_range_m"],
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
    parser.add_argument("--pace", type=float, default=0.02, help="seconds per streamed frame")
    parser.add_argument("--render-every", type=int, default=20)
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

    state = LiveState(output_dir, run_uuid, args.seed, backend, gpu)
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

    from echos_frontier import MissionAborted, MissionContactFailure, run_frontier_exploration
    from replay_argus import replay

    renderer = LiveRenderer(state, SOUND_SOURCE_WORLD, args.pace)
    DashboardHandler.renderer = renderer
    renderer.reset_view()

    last_phase = [None]

    def on_event(event):
        if event["phase"] != last_phase[0]:
            state.log(f"EVENT phase={event['phase']} step={event['step']}")
            last_phase[0] = event["phase"]
        renderer.render(event)

    trajectory_path = output_dir / "frontier_trajectory.npz"
    def mission_worker():
      try:
        mission = run_frontier_exploration(record_path=trajectory_path,
                                            live_callback=on_event,
                                            live_interval=args.render_every,
                                            step_guard=state.step_guard)
        state.log("REPLAY ARGUS starting")
        estimate, truth = replay(str(trajectory_path), target=SOUND_SOURCE_WORLD)
        error = float(np.linalg.norm(estimate - truth))
        report = write_artifacts(output_dir, mission, estimate, truth, error,
                                 state.run_uuid, args.seed, backend, gpu)
        final_event = {
            "phase": "COMPLETE", "step": mission["term_step"] + 21 + len(mission["rtl_trajectory"]),
            "sim_time_s": float(renderer.state.telemetry.get("sim_time_s", 0.0)),
            "position": mission["rtl_trajectory"][-1] if len(mission["rtl_trajectory"]) else np.array([0.0, 0.0, 1.0]),
            "coverage_percent": mission["coverage_percent"],
            "collision_count": mission["collision_count"],
            "physical_clearance_m": mission["min_clearance"],
            "sensor_range_m": mission["sensor_range_m"],
            "active_contact_count": 0, "contact_count": mission["collision_count"],
            "last_contact": None, "safety_stop_active": False,
            "unknown_traversal": mission["unknown_traversal"],
            "frontiers_discovered": len(mission["frontiers"]),
            "selected_frontier": None, "rtl_distance_m": mission["rtl_dist"],
            "exploration_path": mission["trajectory"], "rtl_path": mission["rtl_trajectory"],
        }
        renderer.render(final_event, estimate=estimate, error=error, gates=report)
        state.finish(report)
        state.log(f"FINAL_JSON {output_dir / 'final.json'}")
        if not report["mission_passed"]:
            state.log("MISSION FAIL final dashboard retained")
            return
        state.log("SERVER_RETAINED final dashboard is available until process exit")
      except MissionContactFailure as exc:
        report = {"run_uuid": state.run_uuid, "seed": args.seed, "backend": backend,
                  "gpu": gpu, "contact_failure": True, "mission_passed": False,
                  "contact_step": exc.step, "contacts": exc.contacts,
                  "physical_clearance_m": exc.physical_clearance_m,
                  "sensor_range_m": state.telemetry.get("sensor_range_m"),
                  "acceptance_gates": {"no_physical_contact": False,
                                        "physical_clearance_gt_0_20_m": exc.physical_clearance_m > 0.20}}
        (output_dir / "final.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        state.finish(report)
        state.log("CONTACT_FAIL final dashboard retained")
      except MissionAborted:
        report = {"run_uuid": state.run_uuid, "seed": args.seed, "backend": backend,
                  "gpu": gpu, "aborted": True, "mission_passed": False,
                  "acceptance_gates": {}}
        state.finish(report)
        state.log("ABORTED final dashboard retained")
      except Exception as exc:
        state.log(f"FAIL {type(exc).__name__}: {exc}")

    state.start_callback = lambda: threading.Thread(target=mission_worker, daemon=True).start()

    def paused_heartbeat():
        while not state.shutdown_event.wait(0.5):
            if state.started and state.paused and not state.done:
                renderer.render_heartbeat()

    threading.Thread(target=paused_heartbeat, daemon=True).start()
    try:
        while not state.shutdown_event.wait(1.0):
            pass
    except KeyboardInterrupt:
        state.log("SERVER_STOP requested")
    finally:
        server.shutdown()
        state.close()


if __name__ == "__main__":
    main()
