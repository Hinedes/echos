"""GPU frontier mission -> ARGUS sound localization -> heatmap -> RTL."""

import json
import os
import sys

import numpy as np

from echos_frontier import run_frontier_exploration
from replay_argus import replay


SOUND_SOURCE_WORLD = np.array([2.4, 0.0, 1.0])


def _write_pgm(path, image):
    image = np.asarray(image, dtype=float)
    lo, hi = float(image.min()), float(image.max())
    pixels = np.zeros(image.shape, dtype=np.uint8) if hi <= lo else (
        255.0 * (image - lo) / (hi - lo)).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
        f.write(pixels.tobytes())


def _map_xy(mapper, point, width=800, height=600):
    x = (point[0] - mapper.x_min) / (mapper.x_max - mapper.x_min) * width
    y = height - (point[1] - mapper.y_min) / (mapper.y_max - mapper.y_min) * height
    return x, y


def _write_svg(path, mapper, occupancy, trajectory, rtl_trajectory,
               truth, estimate):
    width, height = 800, 600
    cells = []
    for iy in range(occupancy.shape[0]):
        for ix in range(occupancy.shape[1]):
            value = occupancy[iy, ix]
            color = "#202020" if value == 0.0 else "#e8f0e8" if value == 0.5 else "#8b3030"
            x, y = _map_xy(mapper, mapper.g2w(ix, iy), width, height)
            cw = width / mapper.w + 0.5
            ch = height / mapper.h + 0.5
            cells.append(f'<rect x="{x:.2f}" y="{y - ch:.2f}" width="{cw:.2f}" height="{ch:.2f}" fill="{color}"/>')

    def line(points, color, width_px):
        if len(points) == 0:
            return ""
        coords = " ".join(f"{_map_xy(mapper, p) [0]:.2f},{_map_xy(mapper, p)[1]:.2f}" for p in points)
        return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width_px}"/>'

    tx, ty = _map_xy(mapper, truth, width, height)
    ex, ey = _map_xy(mapper, estimate, width, height)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    svg.extend(cells)
    svg.append(line(trajectory[:, :2], "#28a745", 2))
    svg.append(line(rtl_trajectory[:, :2], "#ff9f1c", 2))
    svg.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="7" fill="none" stroke="#0066ff" stroke-width="3"/>')
    svg.append(f'<circle cx="{ex:.2f}" cy="{ey:.2f}" r="6" fill="#ff00aa"/>')
    svg.append('<text x="12" y="22" fill="white">green: exploration, orange: RTL, blue: truth, magenta: ARGUS</text>')
    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def run(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    trajectory_path = os.path.join(output_dir, "frontier_trajectory.npz")
    mission = run_frontier_exploration(record_path=trajectory_path)
    estimate, truth = replay(trajectory_path, target=SOUND_SOURCE_WORLD)
    error = float(np.linalg.norm(estimate - truth))

    mapper = mission["mapper"]
    occupancy = mapper.get_map()
    heatmap = np.zeros_like(occupancy, dtype=float)
    ix, iy = mapper.w2g(estimate[0], estimate[1])
    if mapper.in_b(ix, iy):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if mapper.in_b(ix + dx, iy + dy):
                    heatmap[iy + dy, ix + dx] = np.exp(-0.5 * (dx * dx + dy * dy))

    np.savez_compressed(
        os.path.join(output_dir, "frontier_maps.npz"),
        occupancy=occupancy, views=mapper.views, sound_heatmap=heatmap,
        trajectory=mission["trajectory"], rtl_trajectory=mission["rtl_trajectory"],
        source_truth=truth, source_estimate=estimate,
    )
    _write_pgm(os.path.join(output_dir, "occupancy_map.pgm"), occupancy)
    _write_pgm(os.path.join(output_dir, "sound_heatmap.pgm"), heatmap)
    _write_svg(
        os.path.join(output_dir, "frontier_argus_demo.svg"), mapper, occupancy,
        mission["trajectory"], mission["rtl_trajectory"], truth, estimate,
    )
    report = {
        "mission_passed": bool(mission["passed"]),
        "terminated_no_frontier": bool(mission["terminated_no_frontier"]),
        "coverage_percent": float(mission["coverage_percent"]),
        "collision": bool(mission["collision"]),
        "min_clearance_m": mission["min_clearance"],
        "unknown_traversal": mission["unknown_traversal"],
        "frontier_oscillation": mission["oscillation"],
        "frontier_centers": [
            [float(frontier["cx"]), float(frontier["cy"])]
            for frontier in mission["frontiers"]
        ],
        "rtl_distance_m": float(mission["rtl_dist"]),
        "source_truth_world_m": truth.tolist(),
        "source_estimate_world_m": estimate.tolist(),
        "localization_error_m": error,
        "trajectory_path": trajectory_path,
    }
    with open(os.path.join(output_dir, "frontier_argus_demo.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not mission["passed"] or error >= 0.1:
        raise RuntimeError("frontier ARGUS demo acceptance failed")
    return report


if __name__ == "__main__":
    destination = sys.argv[1] if len(sys.argv) == 2 else "/workspace/frontier_argus_demo"
    run(destination)
