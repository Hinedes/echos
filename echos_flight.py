"""Flight and ARGUS validation suite with strict disturbance gates.

The original simulation harness is preserved in ``echos_flight_legacy.py``.
This module re-exports it and replaces the two disturbance tests whose legacy
metrics could report false passes.
"""
import numpy as np

from echos_flight_legacy import *
from echos_flight_metrics import disturbance_peak, sustained_recovery_time


def _controlled_thrust(target, pos, vel, q_cur, omega_world):
    total_thrust, q_des = position_pd(target, pos, vel)
    torque, _ = attitude_pd(q_des, q_cur, omega_world)
    thrusts, _, _, raw = mixer_with_authority(total_thrust, *torque)
    has_negative_raw = bool(np.any(raw < -1e-9))
    return thrusts, q_des, has_negative_raw


def run_hover_gust_test():
    reset_body(pos=(0.0, 0.0, 1.0))
    target = np.array([0.0, 0.0, 1.0])
    gust_start = 50
    gust_end = 100
    gust_force = np.array([0.30, 0.0, 0.0])
    total_steps = 800
    max_tilt_allowed = 25.0

    print(f"\n{'='*50}")
    print("  Hover Gust Test")
    print(f"{'='*50}")
    print(
        f"  hover at (0, 0, 1), +0.30 N X from step "
        f"{gust_start}–{gust_end - 1}"
    )

    has_negative = False
    max_tilt = 0.0
    max_deviation = 0.0
    collision = False
    gust_samples = 0
    pos_errors = []

    for step in range(total_steps):
        rigid_solver.clear_external_force()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_world = body.get_ang().cpu().numpy()

        thrusts, _, negative_raw = _controlled_thrust(
            target, pos, vel, q_cur, omega_world
        )
        has_negative |= negative_raw
        apply_rotor_forces(rigid_solver, link_idx, thrusts)

        if gust_start <= step < gust_end:
            gust_samples += 1
            rigid_solver.apply_links_external_force(
                force=gust_force.reshape(1, 3),
                links_idx=[link_idx],
                ref="link_origin",
                local=False,
            )

        scene.step()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        collision |= check_collision(body)
        max_tilt = max(max_tilt, tilt_deg(q_cur))
        pos_error = float(np.linalg.norm(pos - target))
        pos_errors.append(pos_error)
        max_deviation = max(max_deviation, pos_error)

        if collision:
            break

    recovery_time = sustained_recovery_time(
        pos_errors,
        gust_end,
        threshold=0.05,
        hold_steps=20,
        dt=0.01,
    )
    final_pos = body.get_pos().cpu().numpy()
    final_pos_error = float(np.linalg.norm(final_pos - target))
    final_yaw = quat_to_euler(body.get_quat().cpu().numpy())[2]

    passed = all(
        [
            gust_samples == gust_end - gust_start,
            not collision,
            max_tilt < max_tilt_allowed,
            max_deviation < 0.50,
            recovery_time is not None and recovery_time < 3.0,
            final_pos_error < 0.02,
            not has_negative,
        ]
    )

    recovery_text = (
        f"{recovery_time:.2f} s" if recovery_time is not None else "NOT RECOVERED"
    )
    print("\n  Results:")
    print(f"    gust_samples: {gust_samples}")
    print(f"    max_deviation: {max_deviation:.4f} m")
    print(f"    max_tilt: {max_tilt:.2f} deg")
    print(f"    recovery_time: {recovery_text}")
    print(f"    collision: {collision}")
    print(f"    final_pos_err: {final_pos_error:.4f} m")
    print(f"    final_yaw: {final_yaw:.2f} deg")
    print(f"    has_negative_raw: {has_negative}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def run_route_gust_test(waypoints, max_steps_per_leg=2000):
    reset_body(pos=waypoints[0])
    print(f"\n{'='*50}")
    print("  Route Gust Test")
    print(f"{'='*50}")
    print(f"  Route: {waypoints}")

    gust_leg = 2
    gust_start = 50
    gust_end = 100
    gust_force = np.array([0.0, -0.30, 0.0])
    print(
        f"  Leg {gust_leg}: -0.30 N Y gust from step "
        f"{gust_start} to {gust_end - 1}"
    )

    max_waypoint_error = 0.0
    max_tilt = 0.0
    has_negative = False
    yaw_stays_zero = True
    completed_all = True
    collision = False
    total_steps = 0
    yaw_threshold_deg = 1.0
    gust_samples = 0
    gust_errors = []

    for waypoint_index, waypoint in enumerate(waypoints):
        waypoint = np.asarray(waypoint, dtype=float)
        leg_peak = 0.0
        hold_count = 0
        reached = False
        tracking = False

        for step in range(max_steps_per_leg):
            rigid_solver.clear_external_force()
            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            omega_world = body.get_ang().cpu().numpy()

            thrusts, _, negative_raw = _controlled_thrust(
                waypoint, pos, vel, q_cur, omega_world
            )
            has_negative |= negative_raw
            apply_rotor_forces(rigid_solver, link_idx, thrusts)

            if waypoint_index == gust_leg and gust_start <= step < gust_end:
                gust_samples += 1
                rigid_solver.apply_links_external_force(
                    force=gust_force.reshape(1, 3),
                    links_idx=[link_idx],
                    ref="link_origin",
                    local=False,
                )

            scene.step()
            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            collision |= check_collision(body)

            tilt = tilt_deg(q_cur)
            max_tilt = max(max_tilt, tilt)
            pos_error = float(np.linalg.norm(pos - waypoint))
            if not tracking and pos_error < 0.10:
                tracking = True
            if tracking:
                leg_peak = max(leg_peak, pos_error)
                max_waypoint_error = max(max_waypoint_error, pos_error)

            if waypoint_index == gust_leg:
                gust_errors.append(pos_error)

            yaw_deg = quat_to_euler(q_cur)[2]
            if abs(yaw_deg) > yaw_threshold_deg:
                yaw_stays_zero = False

            speed = float(np.linalg.norm(vel))
            hold_count = hold_count + 1 if pos_error < 0.05 and speed < 0.1 else 0
            if hold_count >= 20:
                reached = True
                total_steps += step + 1
                print(
                    f"  Leg {waypoint_index}: → "
                    f"({waypoint[0]:.1f},{waypoint[1]:.1f},{waypoint[2]:.1f}) "
                    f"peak_err={leg_peak:.4f} steps={step + 1:4d} "
                    f"yaw={yaw_deg:+.1f}° OK"
                )
                break
            if collision:
                break

        if not reached:
            completed_all = False
            total_steps += min(max_steps_per_leg, step + 1)
            print(
                f"  Leg {waypoint_index}: → "
                f"({waypoint[0]:.1f},{waypoint[1]:.1f},{waypoint[2]:.1f}) "
                f"peak_err={leg_peak:.4f} TIMEOUT"
            )
        if collision:
            break

    gust_peak = disturbance_peak(gust_errors, gust_start)
    recovery_time = sustained_recovery_time(
        gust_errors,
        gust_end,
        threshold=0.05,
        hold_steps=20,
        dt=0.01,
    )
    final_pos = body.get_pos().cpu().numpy()
    final_pos_error = float(
        np.linalg.norm(final_pos - np.asarray(waypoints[-1], dtype=float))
    )
    final_yaw = quat_to_euler(body.get_quat().cpu().numpy())[2]
    max_tilt_allowed = 25.0

    effective_peak = max(
        max_waypoint_error,
        gust_peak if gust_peak is not None else float("inf"),
    )
    passed = all(
        [
            completed_all,
            gust_samples == gust_end - gust_start,
            gust_peak is not None,
            recovery_time is not None and recovery_time < 3.0,
            effective_peak < 0.50,
            max_tilt < max_tilt_allowed,
            not collision,
            not has_negative,
            final_pos_error < 0.02,
            yaw_stays_zero,
        ]
    )

    recovery_text = (
        f"{recovery_time:.2f} s" if recovery_time is not None else "NOT RECOVERED"
    )
    peak_text = f"{gust_peak:.4f} m" if gust_peak is not None else "NOT MEASURED"
    print("\n  Results:")
    print(f"    completed: {completed_all}")
    print(f"    gust_samples: {gust_samples}")
    print(f"    gust_peak_deviation: {peak_text}")
    print(f"    max_waypoint_error: {max_waypoint_error:.4f} m")
    print(f"    max_tilt: {max_tilt:.2f} deg")
    print(f"    recovery_time: {recovery_text}")
    print(f"    collision: {collision}")
    print(f"    final_pos_err: {final_pos_error:.4f} m")
    print(f"    final_yaw: {final_yaw:.2f} deg")
    print(f"    yaw_stays_zero: {yaw_stays_zero}")
    print(f"    has_negative_raw: {has_negative}")
    print(f"    total_steps: {total_steps}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = []
    results.append(
        run_pos_test(
            "X offset 0.5m",
            init_pos=(0.5, 0.0, 1.0),
            pos_des=(0.0, 0.0, 1.0),
        )
    )
    results.append(
        run_pos_test(
            "Y offset 0.5m",
            init_pos=(0.0, 0.5, 1.0),
            pos_des=(0.0, 0.0, 1.0),
        )
    )
    results.append(
        run_pos_test(
            "Z offset 0.5m",
            init_pos=(0.0, 0.0, 1.5),
            pos_des=(0.0, 0.0, 1.0),
        )
    )
    route = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    results.append(run_waypoint_mission(route))
    results.append(run_hover_gust_test())
    results.append(run_route_gust_test(route))
    results.append(run_yaw_error_test())
    results.append(run_argus_tests())
    results.append(run_scan_tests())
    results.append(run_motion_scan())
    results.append(run_3d_stationary_scan())
    raise SystemExit(0 if all(results) else 1)
