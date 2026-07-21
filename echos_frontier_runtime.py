"""Frontier exploration mission runtime."""
import hashlib

import numpy as np

from echos_core import *
from echos_frontier_logic import *
from echos_frontier_scene import _build_scene, _scan_into_map


def run_frontier_exploration(max_steps=15000, rtl_max_steps=8000):
    gs, scene, body, flood, throw, rigid_solver, link_idx = _build_scene()
    try:
        print("\n" + "=" * 50)
        print("  Autonomous Frontier Exploration")
        print("=" * 50)
        clear_failed_frontiers()

        reset_body_state(body, rigid_solver, pos=(0.0, 0.0, 1.0))
        for _ in range(20):
            rigid_solver.clear_external_force()
            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            omega = body.get_ang().cpu().numpy()
            thrust, q_des = position_pd(np.array([0.0, 0.0, 1.0]), pos, vel)
            torque, _ = attitude_pd(q_des, q_cur, omega)
            rotor_thrusts, _, _, _ = mixer_with_authority(thrust, *torque)
            apply_rotor_forces(rigid_solver, link_idx, rotor_thrusts)
            scene.step()

        mapper = OccupancyMapper(MAP_BOUNDS, MAP_RESOLUTION)
        az_deg = np.linspace(-30.0, 30.0, 7)
        launch = np.array([0.0, 0.0, 1.0])
        bootstrap = bootstrap_target(launch)
        replan_interval = 120

        state = "INITIAL_FLY"
        path = []
        waypoint_i = 0
        trajectory = []
        frontier_log = []
        decisions = []
        min_clearance = float("inf")
        collided = False
        best = None
        unknown_traversal = 0
        yaw_cmd = 0.0
        hold_pos = None
        align_steps = 0
        scan_steps = 0
        yaw_aligned = False
        term_step = max_steps
        pos = launch.copy()

        for step in range(max_steps):
            rigid_solver.clear_external_force()
            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            omega = body.get_ang().cpu().numpy()

            raw_flood = _scan_into_map(
                mapper, flood, throw, q_cur, pos, az_deg, step
            )
            valid = raw_flood[raw_flood >= 0.0]
            if valid.size:
                min_clearance = min(min_clearance, float(np.min(valid)))

            if step >= 300 and state == "INITIAL_FLY":
                state = "EXPLORE"

            if step >= 300 and state == "EXPLORE" and step % replan_interval == 0:
                occ = mapper.get_map()
                inflated = inflation_grid(mapper, inflate_r=4)
                clusters = frontier_clusters(occ, inflated)
                best, path = select_frontier(occ, inflated, clusters, pos, mapper)
                if best is None:
                    decisions.append(
                        (step, "NO_FRONTIER", float(pos[0]), float(pos[1]))
                    )
                    print(f"  step {step}: no reachable frontier -> RTL")
                    term_step = step
                    break
                state = "FLY_TO_FRONTIER"
                waypoint_i = 0
                frontier_log.append(
                    {
                        "step": step,
                        "cx": best["cx"],
                        "cy": best["cy"],
                        "n": best["n"],
                        "score": best["score"],
                        "observe_heading": best["observe_heading"],
                    }
                )
                if len(frontier_log) <= 10:
                    print(
                        f"  step {step}: frontier #{len(frontier_log)} "
                        f"({best['cx']:.2f},{best['cy']:.2f}) "
                        f"n={best['n']} score={best['score']:.3f}"
                    )

            if state == "HOLD_AND_OBSERVE":
                yaw_cmd = move_toward_angle(
                    yaw_cmd,
                    best["observe_heading"],
                    MAX_YAW_RATE * 0.01,
                )
                if not yaw_aligned and align_steps < 200:
                    align_steps += 1
                    body_x = R_world_from_body(q_cur) @ np.array([1.0, 0.0, 0.0])
                    unknown_x, unknown_y = best["uk_world"]
                    unknown_dir = np.array(
                        [unknown_x - best["cx"], unknown_y - best["cy"], 0.0]
                    )
                    unknown_dir /= max(np.linalg.norm(unknown_dir), 1e-10)
                    if body_x @ unknown_dir > 0.99:
                        yaw_aligned = True
                elif yaw_aligned and scan_steps < 30:
                    scan_steps += 1
                elif yaw_aligned:
                    state = "EXPLORE"
                else:
                    mark_frontier_failed(best["ccx"], best["ccy"], mapper)
                    state = "EXPLORE"
                    print(
                        f"    ALIGN TIMEOUT at step {step} — frontier cooled down"
                    )

            if step < 300:
                target = bootstrap.copy()
            elif state == "FLY_TO_FRONTIER" and waypoint_i < len(path):
                wx, wy = mapper.g2w(*path[waypoint_i])
                target = np.array([wx, wy, 1.0])
                if (
                    np.linalg.norm(pos[:2] - target[:2]) < mapper.res * 1.5
                    and np.linalg.norm(vel[:2]) < 0.3
                ):
                    waypoint_i += 1
            elif state == "FLY_TO_FRONTIER":
                target = np.array([pos[0], pos[1], 1.0])
                state = "HOLD_AND_OBSERVE"
                hold_pos = pos.copy()
                hold_pos[2] = 1.0
                align_steps = 0
                scan_steps = 0
                yaw_aligned = False
            elif state == "HOLD_AND_OBSERVE":
                target = hold_pos.copy()
            else:
                target = np.array([pos[0], pos[1], 1.0])

            thrust, q_des = position_pd(target, pos, vel, yaw_cmd)
            torque, _ = attitude_pd(q_des, q_cur, omega)
            rotor_thrusts, _, _, _ = mixer_with_authority(thrust, *torque)
            apply_rotor_forces(rigid_solver, link_idx, rotor_thrusts)
            scene.step()

            pos = body.get_pos().cpu().numpy()
            trajectory.append(pos.copy())
            if step >= 300 and not cell_is_known_free(mapper, pos):
                unknown_traversal += 1
            if check_collision(body):
                collided = True
                term_step = step
                break

        inflated = inflation_grid(mapper, inflate_r=4)
        start = mapper.w2g(pos[0], pos[1])
        goal = mapper.w2g(launch[0], launch[1])
        start = (
            max(0, min(mapper.w - 1, start[0])),
            max(0, min(mapper.h - 1, start[1])),
        )
        goal = (
            max(0, min(mapper.w - 1, goal[0])),
            max(0, min(mapper.h - 1, goal[1])),
        )
        rtl_path = None if collided else astar_path(inflated, start, goal)
        final_speed = float("inf")

        if rtl_path is None:
            print(f"  RTL FAILED: no path from {start} to {goal}")
        else:
            waypoint_i = 0
            for rtl_step in range(rtl_max_steps):
                rigid_solver.clear_external_force()
                q_cur = body.get_quat().cpu().numpy()
                pos = body.get_pos().cpu().numpy()
                vel = body.get_vel().cpu().numpy()
                omega = body.get_ang().cpu().numpy()

                if waypoint_i >= len(rtl_path):
                    target = launch.copy()
                else:
                    wx, wy = mapper.g2w(*rtl_path[waypoint_i])
                    target = np.array([wx, wy, 1.0])
                if (
                    np.linalg.norm(pos[:2] - target[:2]) < mapper.res * 1.5
                    and np.linalg.norm(vel[:2]) < 0.3
                ):
                    waypoint_i += 1

                thrust, q_des = position_pd(target, pos, vel, yaw_cmd)
                torque, _ = attitude_pd(q_des, q_cur, omega)
                rotor_thrusts, _, _, _ = mixer_with_authority(thrust, *torque)
                apply_rotor_forces(rigid_solver, link_idx, rotor_thrusts)
                scene.step()

                pos = body.get_pos().cpu().numpy()
                vel = body.get_vel().cpu().numpy()
                trajectory.append(pos.copy())
                if not cell_is_known_free(mapper, pos):
                    unknown_traversal += 1
                if check_collision(body):
                    collided = True
                    break

                flood_data = flood.read().distances.cpu().numpy().flatten()
                valid = flood_data[flood_data >= 0.0]
                if valid.size:
                    min_clearance = min(min_clearance, float(np.min(valid)))
                final_speed = float(np.linalg.norm(vel[:2]))
                if (
                    np.linalg.norm(pos[:2] - launch[:2]) < 0.04
                    and final_speed < 0.1
                    and rtl_step > 50
                ):
                    break

        final_pos = body.get_pos().cpu().numpy()
        rtl_distance = float(np.linalg.norm(final_pos[:2] - launch[:2]))
        (
            coverage,
            horizontal_seen,
            vertical_seen,
            reachable_n,
            discovered_n,
        ) = exploration_metrics(mapper, launch)
        oscillation = False
        if len(frontier_log) >= 3:
            recent = [
                (round(item["cx"], 3), round(item["cy"], 3))
                for item in frontier_log[-5:]
            ]
            oscillation = len(set(recent)) < len(recent)

        print(
            f"\n  Trajectory: {len(trajectory)} steps, "
            f"final ({final_pos[0]:.2f},{final_pos[1]:.2f})"
        )
        print(
            f"  Frontiers: {len(frontier_log)}, explored {coverage:.1f}% "
            f"({discovered_n}/{reachable_n})"
        )
        print(f"  Both legs: horiz={horizontal_seen} vert={vertical_seen}")
        print(f"  Unknown traversal: {unknown_traversal} steps")
        print(f"  Oscillation: {oscillation}")
        print(
            f"  RTL dist: {rtl_distance:.4f}, speed: {final_speed:.4f}, "
            f"collision: {collided}"
        )
        print(f"  Min clearance: {min_clearance:.4f}")

        criteria = {
            "coverage": coverage >= 95.0,
            "no_collision": not collided,
            "clearance": np.isfinite(min_clearance) and min_clearance > 0.20,
            "frontiers_found": len(frontier_log) > 0,
            "rtl": (
                rtl_path is not None
                and rtl_distance < 0.05
                and final_speed < 0.1
            ),
            "both_legs": horizontal_seen and vertical_seen,
            "known_only": unknown_traversal == 0,
            "no_oscillation": not oscillation,
            "no_frontier_termination": any(
                item[1] == "NO_FRONTIER" for item in decisions
            ),
        }
        passed = all(criteria.values())
        print("\n" + "=" * 50)
        print("  Pass Criteria")
        print("=" * 50)
        for name, result in criteria.items():
            print(f"  {name}: {'PASS' if result else 'FAIL'}")
        print(f"\n  FRONTIER EXPLORATION {'PASS' if passed else 'FAIL'}")

        trajectory_array = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
        frontier_array = np.asarray(
            [(item["cx"], item["cy"]) for item in frontier_log],
            dtype=np.float64,
        ).reshape(-1, 2)
        return {
            "passed": passed,
            "traj_hash": hashlib.sha256(
                trajectory_array.tobytes()
            ).hexdigest(),
            "frontier_seq_hash": hashlib.sha256(
                frontier_array.tobytes()
            ).hexdigest(),
            "occ_grid_hash": hashlib.sha256(
                mapper.get_map().tobytes()
            ).hexdigest(),
            "term_step": term_step,
            "rtl_dist": rtl_distance,
            "coverage": coverage,
            "unknown_traversal": unknown_traversal,
            "criteria": criteria,
        }
    finally:
        gs.destroy()


def verify_frontier_determinism():
    first = run_frontier_exploration()
    second = run_frontier_exploration()
    hashes_match = all(
        first[key] == second[key]
        for key in ("traj_hash", "frontier_seq_hash", "occ_grid_hash")
    )
    metrics_match = (
        first["term_step"] == second["term_step"]
        and first["coverage"] == second["coverage"]
        and first["unknown_traversal"] == second["unknown_traversal"]
    )
    passed = (
        first["passed"]
        and second["passed"]
        and hashes_match
        and metrics_match
    )
    print(
        f"Deterministic frontier exploration: "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed
