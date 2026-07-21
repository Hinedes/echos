"""Occupancy mapping and map-driven RTL with strict safety validation.

The original Genesis scene is preserved in ``echos_mapper_legacy.py``. This
module reuses that exact scene and replaces the mission evaluator so collisions,
missing measurements, and non-settled RTL cannot report success.
"""
import numpy as np

from echos_mapper_legacy import *


def _finite_clearance(value, threshold=0.20):
    return bool(np.isfinite(value) and value > threshold)


def run_mapping_mission(max_steps=6000, rtl_max_steps=8000):
    print(f"\n{'='*50}")
    print("  2D Occupancy Grid Mapping")
    print(f"{'='*50}")

    reset_body_state(body, rs, pos=(0.0, 0.0, 1.0))
    for _ in range(20):
        rs.clear_external_force()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega = body.get_ang().cpu().numpy()
        total_thrust, q_des = position_pd(np.array([0.0, 0.0, 1.0]), pos, vel)
        torque, _ = attitude_pd(q_des, q_cur, omega)
        thrusts, _, _, _ = mixer_with_authority(total_thrust, *torque)
        apply_rotor_forces(rs, li, thrusts)
        scene.step()

    mapper = OccupancyMapper((-2.0, 7.0, -2.0, 5.0), resolution=0.08)
    azimuths = np.linspace(-30.0, 30.0, 7)
    waypoints = [np.array([5.0, 0.0, 1.0]), np.array([5.0, 2.5, 1.0])]
    waypoint_index = 0
    trajectory = []
    ray_log = []
    mapping_collision = False
    mapping_min_clearance = float("inf")

    for step in range(max_steps):
        rs.clear_external_force()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega = body.get_ang().cpu().numpy()

        flood_data = argus_flood.read()
        raw_flood = flood_data.distances.cpu().numpy().flatten()
        valid_flood = raw_flood[raw_flood >= 0.0]
        if valid_flood.size:
            mapping_min_clearance = min(
                mapping_min_clearance, float(np.min(valid_flood))
            )

        use_throw = step % 5 == 0
        raw_throw = None
        if use_throw:
            raw_throw = argus_throw.read().distances.flatten()[0].item()

        Rwb = R_world_from_body(q_cur)
        emitter_world = pos + Rwb @ np.array([RAYCAST_ORIGIN, 0.0, 0.0])
        for ray_index, azimuth_deg in enumerate(azimuths):
            azimuth = np.radians(azimuth_deg)
            direction_body = np.array(
                [np.cos(azimuth), np.sin(azimuth), 0.0]
            )
            direction_world = Rwb @ direction_body
            hit = raw_flood[ray_index] >= 0.0
            ray_range = raw_flood[ray_index] if hit else 0.0
            mapper.update_ray(
                emitter_world, direction_world, ray_range, 5.0, hit
            )
            ray_log.append(
                (
                    step,
                    "FLOOD",
                    ray_index,
                    azimuth_deg,
                    pos.copy(),
                    q_cur.copy(),
                    ray_range if hit else -1.0,
                    hit,
                )
            )

        if use_throw:
            direction_world = Rwb @ np.array([1.0, 0.0, 0.0])
            hit = raw_throw >= 0.0
            mapper.update_ray(
                emitter_world,
                direction_world,
                raw_throw if hit else 0.0,
                8.0,
                hit,
            )
            ray_log.append(
                (
                    step,
                    "THROW",
                    0,
                    0.0,
                    pos.copy(),
                    q_cur.copy(),
                    raw_throw if hit else -1.0,
                    hit,
                )
            )

        target = waypoints[min(waypoint_index, len(waypoints) - 1)].copy()
        if np.linalg.norm(pos[:2] - target[:2]) < 0.25:
            waypoint_index = min(waypoint_index + 1, len(waypoints))

        total_thrust, q_des = position_pd(target, pos, vel)
        torque, _ = attitude_pd(q_des, q_cur, omega)
        thrusts, _, _, _ = mixer_with_authority(total_thrust, *torque)
        apply_rotor_forces(rs, li, thrusts)
        scene.step()

        pos = body.get_pos().cpu().numpy()
        trajectory.append(pos.copy())
        if check_collision(body):
            mapping_collision = True
            print(f"  Mapping collision at step {step}")
            break
        if (
            waypoint_index >= len(waypoints)
            and np.linalg.norm(pos[:2] - waypoints[-1][:2]) < 0.3
            and step > 3000
        ):
            break

    final_mapping_pos = body.get_pos().cpu().numpy()
    print(
        f"\n  Trajectory: {len(trajectory)} steps, final "
        f"({final_mapping_pos[0]:.2f}, {final_mapping_pos[1]:.2f})"
    )
    print(f"  FLOOD rays: {sum(1 for ray in ray_log if ray[1] == 'FLOOD')}")
    print(f"  THROW rays: {sum(1 for ray in ray_log if ray[1] == 'THROW')}")
    print(f"  Mapping collision: {mapping_collision}")
    print(f"  Mapping minimum clearance: {mapping_min_clearance:.4f}")

    occ_map = mapper.get_map()
    cell = mapper.res
    gt_ext = cell
    gt_walls = [
        ("bottom", (-0.75, 6.25, -1.5 - gt_ext, -1.5 + gt_ext)),
        ("left", (-0.5 - gt_ext, -0.5 + gt_ext, -2.0, 4.0)),
        ("top", (-0.5, 3.0, 1.5 - gt_ext, 1.5 + gt_ext)),
        ("inner_right", (3.0 - gt_ext, 3.0 + gt_ext, 1.25, 4.25)),
        ("end", (3.0, 6.5, 4.0 - gt_ext, 4.0 + gt_ext)),
        ("obstacle", (1.25, 1.75, 0.6, 1.4)),
    ]
    free_margin = cell * 2.0

    def in_ground_truth_wall(x, y):
        return any(
            x1 <= x <= x2 and y1 <= y <= y2
            for _, (x1, x2, y1, y2) in gt_walls
        )

    def in_ground_truth_free(x, y):
        in_horizontal = (
            -0.5 + free_margin <= x <= 6.25 - free_margin
            and -1.5 + free_margin <= y <= 1.5 - free_margin
        )
        in_vertical = (
            3.0 + free_margin <= x <= 6.5 - free_margin
            and 1.5 + free_margin <= y <= 4.0 - free_margin
        )
        return (in_horizontal or in_vertical) and not in_ground_truth_wall(x, y)

    true_occupied = false_occupied = true_free = false_free = unknown = 0
    for iy in range(mapper.h):
        for ix in range(mapper.w):
            if mapper.views[iy, ix] <= 0:
                continue
            world_x, world_y = mapper.g2w(ix, iy)
            ground_wall = in_ground_truth_wall(world_x, world_y)
            ground_free = in_ground_truth_free(world_x, world_y)
            if ground_wall == ground_free:
                continue

            predicted_occupied = occ_map[iy, ix] == 1.0
            predicted_free = occ_map[iy, ix] == 0.5
            if ground_wall:
                if predicted_occupied:
                    true_occupied += 1
                elif predicted_free:
                    false_free += 1
                else:
                    unknown += 1
            else:
                if predicted_free:
                    true_free += 1
                elif predicted_occupied:
                    false_occupied += 1
                else:
                    unknown += 1

    occupied_precision = (
        100.0 * true_occupied / (true_occupied + false_occupied)
        if true_occupied + false_occupied > 0
        else 0.0
    )
    free_precision = (
        100.0 * true_free / (true_free + false_free)
        if true_free + false_free > 0
        else 0.0
    )
    print(
        f"\n  Confusion: TP={true_occupied} FP={false_occupied} "
        f"TN={true_free} FN={false_free} UNK={unknown}"
    )
    print(f"  Occupied precision: {occupied_precision:.1f}%")
    print(f"  Free precision: {free_precision:.1f}%")

    mapping_criteria = {
        "no_collision": not mapping_collision,
        "finite_clearance": _finite_clearance(mapping_min_clearance),
        "flood_used": any(ray[1] == "FLOOD" for ray in ray_log),
        "throw_used": any(ray[1] == "THROW" for ray in ray_log),
        "occupied_samples": true_occupied + false_occupied > 0,
        "free_samples": true_free + false_free > 0,
        "occupied_precision": occupied_precision > 50.0,
        "free_precision": free_precision > 80.0,
    }
    mapping_passed = all(mapping_criteria.values())

    print(f"\n{'='*50}")
    print("  Mapping Pass Criteria")
    print(f"{'='*50}")
    for name, result in mapping_criteria.items():
        print(f"  {name}: {'PASS' if result else 'FAIL'}")
    print(f"\n  MAPPING MISSION {'PASS' if mapping_passed else 'FAIL'}")

    rtl_passed = False
    print(f"\n{'='*50}")
    print("  Map-Driven Return-to-Launch")
    print(f"{'='*50}")

    if trajectory and not mapping_collision:
        rtl_start = trajectory[-1].copy()
        reset_body_state(
            body, rs, pos=(rtl_start[0], rtl_start[1], 1.0)
        )
        for _ in range(30):
            rs.clear_external_force()
            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            omega = body.get_ang().cpu().numpy()
            hold = np.array([rtl_start[0], rtl_start[1], 1.0])
            total_thrust, q_des = position_pd(hold, pos, vel)
            torque, _ = attitude_pd(q_des, q_cur, omega)
            thrusts, _, _, _ = mixer_with_authority(total_thrust, *torque)
            apply_rotor_forces(rs, li, thrusts)
            scene.step()

        launch = np.array([0.0, 0.0, 1.0])
        current = body.get_pos().cpu().numpy()
        inflated = inflation_grid(mapper, inflate_r=4)
        start = mapper.w2g(current[0], current[1])
        goal = mapper.w2g(launch[0], launch[1])
        start = (
            max(0, min(mapper.w - 1, start[0])),
            max(0, min(mapper.h - 1, start[1])),
        )
        goal = (
            max(0, min(mapper.w - 1, goal[0])),
            max(0, min(mapper.h - 1, goal[1])),
        )
        path = astar_path(inflated, start, goal)
        if path is None:
            print(f"  A* FAILED: no path from {start} to {goal}")
        else:
            waypoint_index = 0
            rtl_collision = False
            rtl_min_clearance = float("inf")
            final_speed = float("inf")
            for rtl_step in range(rtl_max_steps):
                rs.clear_external_force()
                q_cur = body.get_quat().cpu().numpy()
                pos = body.get_pos().cpu().numpy()
                vel = body.get_vel().cpu().numpy()
                omega = body.get_ang().cpu().numpy()

                if waypoint_index >= len(path):
                    target = launch.copy()
                else:
                    world_x, world_y = mapper.g2w(*path[waypoint_index])
                    target = np.array([world_x, world_y, 1.0])
                if (
                    np.linalg.norm(pos[:2] - target[:2]) < mapper.res * 1.5
                    and np.linalg.norm(vel[:2]) < 0.3
                ):
                    waypoint_index += 1

                total_thrust, q_des = position_pd(target, pos, vel)
                torque, _ = attitude_pd(q_des, q_cur, omega)
                thrusts, _, _, _ = mixer_with_authority(total_thrust, *torque)
                apply_rotor_forces(rs, li, thrusts)
                scene.step()

                pos = body.get_pos().cpu().numpy()
                vel = body.get_vel().cpu().numpy()
                rtl_collision |= check_collision(body)
                valid = argus_flood.read().distances.cpu().numpy().flatten()
                valid = valid[valid >= 0.0]
                if valid.size:
                    rtl_min_clearance = min(
                        rtl_min_clearance, float(np.min(valid))
                    )
                final_speed = float(np.linalg.norm(vel[:2]))
                if rtl_collision:
                    break
                if (
                    np.linalg.norm(pos[:2] - launch[:2]) < 0.04
                    and final_speed < 0.1
                    and rtl_step > 50
                ):
                    break

            final_pos = body.get_pos().cpu().numpy()
            rtl_distance = float(np.linalg.norm(final_pos[:2] - launch[:2]))
            rtl_criteria = {
                "path_found": path is not None,
                "no_collision": not rtl_collision,
                "finite_clearance": _finite_clearance(rtl_min_clearance),
                "within_5cm": rtl_distance < 0.05,
                "settled": final_speed < 0.1,
            }
            rtl_passed = all(rtl_criteria.values())
            print(
                f"\n  Return leg: start ({current[0]:.2f},{current[1]:.2f}) "
                f"final ({final_pos[0]:.4f},{final_pos[1]:.4f})"
            )
            print(
                f"    dist={rtl_distance:.4f} speed={final_speed:.4f} "
                f"clear={rtl_min_clearance:.4f} coll={rtl_collision}"
            )
            for name, result in rtl_criteria.items():
                print(f"  {name}: {'PASS' if result else 'FAIL'}")
            print(f"  RETURN-TO-LAUNCH {'PASS' if rtl_passed else 'FAIL'}")
    else:
        print("  RTL skipped because mapping did not produce a safe trajectory")

    return mapping_passed and rtl_passed


if __name__ == "__main__":
    raise SystemExit(0 if run_mapping_mission() else 1)
