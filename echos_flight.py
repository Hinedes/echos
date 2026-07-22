import genesis as gs
import numpy as np
import os
from echos_core import *
from argus_export import GimbalState, TrajectoryRecorder

gs.init(backend=getattr(gs, os.environ.get("ECHOS_GENESIS_BACKEND", "gpu")))

scene = gs.Scene(
    show_viewer=False,
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
    ),
)

# physical constants imported from echos_core

body = scene.add_entity(
    gs.morphs.Box(size=(BODY_W, BODY_D, BODY_H), pos=(0.0, 0.0, 1.0), fixed=False),
)
wall_surface_x = 3.0125  # front-wall surface
wall_surface_y = 3.0125  # side-wall surface
wall = scene.add_entity(
    gs.morphs.Box(size=(0.01, 4.0, 2.0), pos=(wall_surface_x + 0.005, 0.0, 1.0), fixed=True),
)
wall_y = scene.add_entity(
    gs.morphs.Box(size=(4.0, 0.01, 2.0), pos=(0.0, wall_surface_y + 0.005, 1.0), fixed=True),
)

floor = scene.add_entity(
    gs.morphs.Plane(),
)

emitter_offset = (BODY_W / 2, 0.0, 0.0)
pattern_fwd = gs.sensors.SphericalPattern(
    angles=(np.array([0.0]), np.array([0.0])),
)
argus_0deg = scene.add_sensor(
    gs.sensors.Raycaster(
        pattern=pattern_fwd,
        entity_idx=body.idx,
        pos_offset=emitter_offset,
        euler_offset=(0.0, 0.0, 0.0),
        max_range=10.0,
        no_hit_value=-1.0,
    )
)
argus_15deg = scene.add_sensor(
    gs.sensors.Raycaster(
        pattern=pattern_fwd,
        entity_idx=body.idx,
        pos_offset=emitter_offset,
        euler_offset=(0.0, 15.0, 0.0),
        max_range=10.0,
        no_hit_value=-1.0,
    )
)
pitch_angles_deg = np.arange(-30, 31, 5)
n_pitch = len(pitch_angles_deg)
SIM_DT_S = 0.01
scan_pattern = gs.sensors.SphericalPattern(
    fov=(0.0, 60.0),
    n_points=(1, n_pitch),
)
argus_scan = scene.add_sensor(
    gs.sensors.Raycaster(
        pattern=scan_pattern,
        entity_idx=body.idx,
        pos_offset=emitter_offset,
        euler_offset=(0.0, 0.0, 0.0),
        max_range=10.0,
        no_hit_value=-1.0,
    )
)

scene.build()
body.set_mass(MASS)
rigid_solver = scene.sim.rigid_solver
link_idx = 0

# imported from echos_core

# imported from echos_core

# imported from echos_core

# imported from echos_core  (renamed R_world_from_body)

def angle_error(q_des, q_cur):
    q_err = quat_mul(quat_conj(q_des), q_cur)
    return 2.0 * np.arccos(np.clip(abs(q_err[0]), 0.0, 1.0))

def cos_tilt(q):
    return 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])

def tilt_deg(q):
    return np.degrees(np.arccos(np.clip(cos_tilt(q), -1.0, 1.0)))

# imported from echos_core  (renamed quat_from_R)

# imported from echos_core

# imported from echos_core

# imported from echos_core

# imported from echos_core

# imported from echos_core

# imported from echos_core
def reset_body(pos=(0.0, 0.0, 1.0), quat=None):
    reset_body_state(body, rigid_solver, pos, quat)

def run_pos_test(name, init_pos, pos_des, steps=600):
    reset_body(pos=init_pos)
    print(f"\n=== {name} ===")
    print(f"  target: ({pos_des[0]:.2f}, {pos_des[1]:.2f}, {pos_des[2]:.2f})")

    has_negative = False
    max_tilt = 0.0
    settle_step = steps
    pos_errs = []
    att_errs = []
    q_des = np.array([1.0, 0.0, 0.0, 0.0])

    for i in range(steps):
        rigid_solver.clear_external_force()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_w = body.get_ang().cpu().numpy()

        T_total, q_des = position_pd(pos_des, pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)

        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        thrusts = np.clip(thrusts, 0.0, None)

        if np.any(thrusts <= 1e-9):
            has_negative = True

        apply_rotor_forces(rigid_solver, link_idx, thrusts)
        scene.step()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        tilt = tilt_deg(q_cur)
        max_tilt = max(max_tilt, tilt)

        pos_err = np.linalg.norm(pos - pos_des)
        att_err = np.degrees(angle_error(q_des, q_cur))
        pos_errs.append(pos_err)
        att_errs.append(att_err)

        if i in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
            r, p, y = quat_to_euler(q_cur)
            print(f"  step {i:4d}: pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                  f"roll={r:+5.1f} pitch={p:+5.1f} tilt={tilt:.1f}")

    for i in reversed(range(steps)):
        if pos_errs[i] >= 0.02 or att_errs[i] >= 1.0:
            settle_step = i + 1
            break

    final_pos = body.get_pos().cpu().numpy()
    final_pos_err = np.linalg.norm(final_pos - pos_des)
    final_q = body.get_quat().cpu().numpy()
    final_att_err = np.degrees(angle_error(q_des, final_q))

    passed = (
        final_pos_err < 0.02
        and final_att_err < 1.0
        and settle_step < 300
        and not has_negative
        and max_tilt < MAX_TILT_DEG
    )

    print(f"  pos_err: {final_pos_err:.4f} m  |  att_err: {final_att_err:.4f} deg  |  "
          f"max_tilt: {max_tilt:.1f} deg  |  settle: {settle_step} steps  |  neg: {has_negative}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed

def run_waypoint_mission(waypoints, max_steps_per_leg=2000):
    reset_body(pos=waypoints[0])
    print(f"\n{'='*50}")
    print(f"  Waypoint Mission")
    print(f"{'='*50}")
    print(f"  Route: {waypoints}")

    max_waypoint_error = 0.0
    max_tilt = 0.0
    has_negative = False
    yaw_stays_zero = True
    completed_all = True
    total_steps = 0
    yaw_threshold_deg = 1.0
    q_des = np.array([1.0, 0.0, 0.0, 0.0])

    for wp_idx, wp in enumerate(waypoints):
        leg_peak = 0.0
        hold_count = 0
        hold_req = 20
        reached = False
        leg_steps = 0
        tracking = False
        init_dist = np.linalg.norm(body.get_pos().cpu().numpy() - np.array(wp))

        for i in range(max_steps_per_leg):
            rigid_solver.clear_external_force()

            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            omega_w = body.get_ang().cpu().numpy()

            T_total, q_des = position_pd(np.array(wp), pos, vel)
            tau, _ = attitude_pd(q_des, q_cur, omega_w)

            thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
            thrusts = np.clip(thrusts, 0.0, None)

            if np.any(thrusts <= 1e-9):
                has_negative = True

            apply_rotor_forces(rigid_solver, link_idx, thrusts)
            scene.step()

            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()

            tilt = tilt_deg(q_cur)
            max_tilt = max(max_tilt, tilt)

            pos_err = np.linalg.norm(pos - np.array(wp))

            if not tracking and pos_err < 0.10:
                tracking = True
            if tracking:
                leg_peak = max(leg_peak, pos_err)
                max_waypoint_error = max(max_waypoint_error, pos_err)

            r, p, y = quat_to_euler(q_cur)
            if abs(y) > yaw_threshold_deg:
                yaw_stays_zero = False

            speed = np.linalg.norm(vel)

            if pos_err < 0.05 and speed < 0.1:
                hold_count += 1
            else:
                hold_count = 0

            if hold_count >= hold_req:
                reached = True
                leg_steps = i + 1
                total_steps += leg_steps
                _, _, y_final = quat_to_euler(q_cur)
                print(f"  Leg {wp_idx}: → ({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f})  "
                      f"peak_err={leg_peak:.4f}  steps={leg_steps:4d}  "
                      f"yaw={y_final:+.1f}°  {'OK' if reached else 'TIMEOUT'}")
                break

        if not reached:
            completed_all = False
            total_steps += max_steps_per_leg
            print(f"  Leg {wp_idx}: → ({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f})  "
                  f"peak_err={leg_peak:.4f}  TIMEOUT")

    final_pos = body.get_pos().cpu().numpy()
    final_pos_err = np.linalg.norm(final_pos - np.array(waypoints[-1]))
    final_q = body.get_quat().cpu().numpy()
    _, _, final_yaw_deg = quat_to_euler(final_q)

    print(f"\n  Results:")
    print(f"    completed: {completed_all}")
    print(f"    max_waypoint_error: {max_waypoint_error:.4f} m")
    print(f"    max_tilt: {max_tilt:.2f} deg")
    print(f"    final_pos_err: {final_pos_err:.4f} m")
    print(f"    final_yaw: {final_yaw_deg:.2f} deg")
    print(f"    has_negative: {has_negative}")
    print(f"    total_steps: {total_steps}")

    passed = (
        completed_all
        and max_waypoint_error < 0.10
        and max_tilt < MAX_TILT_DEG
        and not has_negative
        and final_pos_err < 0.02
        and yaw_stays_zero
    )
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def run_hover_gust_test():
    reset_body(pos=(0.0, 0.0, 1.0))
    target = np.array([0.0, 0.0, 1.0])
    gust_start = 50
    gust_end = 100
    gust_force = np.array([0.30, 0.0, 0.0])
    total_steps = 800
    MAX_TILT_GUST = 25.0

    print(f"\n{'='*50}")
    print(f"  Hover Gust Test")
    print(f"{'='*50}")
    print(f"  hover at (0, 0, 1), +0.30 N X from step {gust_start}–{gust_end-1}")

    has_negative = False
    max_tilt = 0.0
    max_deviation = 0.0
    ground_contact = False
    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    pos_errs = []

    for i in range(total_steps):
        rigid_solver.clear_external_force()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_w = body.get_ang().cpu().numpy()

        T_total, q_des = position_pd(target, pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)

        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        thrusts = np.clip(thrusts, 0.0, None)

        if np.any(thrusts <= 1e-9):
            has_negative = True

        apply_rotor_forces(rigid_solver, link_idx, thrusts)

        if gust_start <= i < gust_end:
            rigid_solver.apply_links_external_force(
                force=gust_force.reshape(1, 3),
                links_idx=[link_idx],
                ref="link_origin",
                local=False,
            )

        scene.step()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()

        if pos[2] <= 0.0:
            ground_contact = True

        tilt = tilt_deg(q_cur)
        max_tilt = max(max_tilt, tilt)

        pos_err = np.linalg.norm(pos - target)
        pos_errs.append(pos_err)
        max_deviation = max(max_deviation, pos_err)

    recovery_step = total_steps
    for i in reversed(range(gust_end, total_steps)):
        if pos_errs[i] >= 0.05:
            recovery_step = i + 1
            break
    recovery_time = (recovery_step - gust_end) * 0.01

    final_pos = body.get_pos().cpu().numpy()
    final_pos_err = np.linalg.norm(final_pos - target)
    final_q = body.get_quat().cpu().numpy()
    _, _, final_yaw_deg = quat_to_euler(final_q)

    passed = (
        not ground_contact
        and max_tilt < MAX_TILT_GUST
        and max_deviation < 0.50
        and recovery_time < 3.0
        and final_pos_err < 0.02
        and not has_negative
    )

    print(f"\n  Results:")
    print(f"    max_deviation: {max_deviation:.4f} m")
    print(f"    max_tilt: {max_tilt:.2f} deg")
    print(f"    recovery_time: {recovery_time:.2f} s ({recovery_step - gust_end} steps)")
    print(f"    ground_contact: {ground_contact}")
    print(f"    final_pos_err: {final_pos_err:.4f} m")
    print(f"    final_yaw: {final_yaw_deg:.2f} deg")
    print(f"    has_negative: {has_negative}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def run_route_gust_test(waypoints, max_steps_per_leg=2000):
    reset_body(pos=waypoints[0])
    print(f"\n{'='*50}")
    print(f"  Route Gust Test")
    print(f"{'='*50}")
    print(f"  Route: {waypoints}")
    print(f"  Leg 2 (→(1,1,1)): -0.30 N Y gust at step 50 for 50 steps")

    max_waypoint_error = 0.0
    max_tilt = 0.0
    has_negative = False
    yaw_stays_zero = True
    completed_all = True
    total_steps = 0
    yaw_threshold_deg = 1.0
    gust_id = 2
    gust_start = 150
    gust_end = 200
    gust_peak = 0.0
    gust_pos_errs = []
    recovery_time = None

    for wp_idx, wp in enumerate(waypoints):
        leg_peak = 0.0
        hold_count = 0
        hold_req = 20
        reached = False
        leg_steps = 0
        tracking = False
        q_des = np.array([1.0, 0.0, 0.0, 0.0])

        for i in range(max_steps_per_leg):
            rigid_solver.clear_external_force()

            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            omega_w = body.get_ang().cpu().numpy()

            T_total, q_des = position_pd(np.array(wp), pos, vel)
            tau, _ = attitude_pd(q_des, q_cur, omega_w)

            thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
            thrusts = np.clip(thrusts, 0.0, None)

            if np.any(thrusts <= 1e-9):
                has_negative = True

            apply_rotor_forces(rigid_solver, link_idx, thrusts)

            if wp_idx == gust_id and gust_start <= i < gust_end:
                gust_force = np.array([0.0, -0.30, 0.0])
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

            tilt = tilt_deg(q_cur)
            max_tilt = max(max_tilt, tilt)

            pos_err = np.linalg.norm(pos - np.array(wp))

            if not tracking and pos_err < 0.10:
                tracking = True
            if tracking:
                leg_peak = max(leg_peak, pos_err)
                max_waypoint_error = max(max_waypoint_error, pos_err)

            if wp_idx == gust_id:
                gust_pos_errs.append(pos_err)
                if i >= gust_end and pos_err > gust_peak:
                    gust_peak = max(gust_peak, pos_err)

            r, p, y = quat_to_euler(q_cur)
            if abs(y) > yaw_threshold_deg:
                yaw_stays_zero = False

            speed = np.linalg.norm(vel)

            if pos_err < 0.05 and speed < 0.1:
                hold_count += 1
            else:
                hold_count = 0

            if hold_count >= hold_req:
                reached = True
                leg_steps = i + 1
                total_steps += leg_steps
                _, _, y_final = quat_to_euler(q_cur)
                print(f"  Leg {wp_idx}: → ({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f})  "
                      f"peak_err={leg_peak:.4f}  steps={leg_steps:4d}  "
                      f"yaw={y_final:+.1f}°  OK")
                break

        if not reached:
            completed_all = False
            total_steps += max_steps_per_leg
            print(f"  Leg {wp_idx}: → ({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.1f})  "
                  f"peak_err={leg_peak:.4f}  TIMEOUT")

    if len(gust_pos_errs) > gust_end:
        for i in reversed(range(gust_end, len(gust_pos_errs))):
            if gust_pos_errs[i] >= 0.05:
                recovery_step = i + 1
                recovery_time = (recovery_step - gust_end) * 0.01
                break

    final_pos = body.get_pos().cpu().numpy()
    final_pos_err = np.linalg.norm(final_pos - np.array(waypoints[-1]))
    final_q = body.get_quat().cpu().numpy()
    _, _, final_yaw_deg = quat_to_euler(final_q)

    MAX_TILT_GUST = 25.0
    recovery_ok = recovery_time is None or recovery_time < 3.0
    # Use max of tracking peak and gust peak
    effective_peak = max(max_waypoint_error, gust_peak)

    print(f"\n  Results:")
    print(f"    completed: {completed_all}")
    print(f"    gust_peak_deviation: {gust_peak:.4f} m")
    print(f"    max_waypoint_error: {max_waypoint_error:.4f} m")
    print(f"    max_tilt: {max_tilt:.2f} deg")
    print(f"    recovery_time: {recovery_time:.2f}s" if recovery_time is not None else "    recovery_time: N/A")
    print(f"    final_pos_err: {final_pos_err:.4f} m")
    print(f"    final_yaw: {final_yaw_deg:.2f} deg")
    print(f"    has_negative: {has_negative}")

    passed = (
        completed_all
        and effective_peak < 0.50
        and max_tilt < MAX_TILT_GUST
        and not has_negative
        and final_pos_err < 0.02
        and recovery_ok
    )
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def run_yaw_error_test():
    yaw_init = 10.0
    q_init = np.array([np.cos(np.radians(yaw_init/2)), 0.0, 0.0, np.sin(np.radians(yaw_init/2))])
    reset_body(pos=(0.0, 0.0, 1.0), quat=q_init)
    target = np.array([0.0, 0.0, 1.0])
    total_steps = 600
    MAX_TILT_YAW = 25.0

    print(f"\n{'='*50}")
    print(f"  Yaw Error Recovery Test")
    print(f"{'='*50}")
    print(f"  start hover at (0,0,1) with {yaw_init}° yaw error")

    has_negative = False
    max_tilt = 0.0
    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    yaw_errs = []

    for i in range(total_steps):
        rigid_solver.clear_external_force()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_w = body.get_ang().cpu().numpy()

        T_total, q_des = position_pd(target, pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)

        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        thrusts = np.clip(thrusts, 0.0, None)

        if np.any(thrusts <= 1e-9):
            has_negative = True

        apply_rotor_forces(rigid_solver, link_idx, thrusts)
        scene.step()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()

        tilt = tilt_deg(q_cur)
        max_tilt = max(max_tilt, tilt)

        _, _, yaw_deg = quat_to_euler(q_cur)
        yaw_errs.append(abs(yaw_deg))

        if i in (0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1):
            r, p, y = quat_to_euler(q_cur)
            print(f"  step {i:4d}: pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                  f"roll={r:+5.1f} pitch={p:+5.1f} yaw={y:+5.1f} tilt={tilt:.1f}")

    final_pos = body.get_pos().cpu().numpy()
    final_pos_err = np.linalg.norm(final_pos - target)
    final_q = body.get_quat().cpu().numpy()
    _, _, final_yaw_deg = quat_to_euler(final_q)

    settle_yaw_step = total_steps
    for i in reversed(range(total_steps)):
        if abs(yaw_errs[i]) > 0.1:
            settle_yaw_step = i + 1
            break

    passed = (
        final_pos_err < 0.02
        and abs(final_yaw_deg) < 0.1
        and settle_yaw_step < 300
        and max_tilt < MAX_TILT_YAW
        and not has_negative
    )

    print(f"\n  Results:")
    print(f"    final_pos_err: {final_pos_err:.4f} m")
    print(f"    final_yaw: {final_yaw_deg:.4f} deg")
    print(f"    yaw_settle: {settle_yaw_step} steps")
    print(f"    max_tilt: {max_tilt:.2f} deg")
    print(f"    has_negative: {has_negative}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def read_argus(sensor, settle_steps=5):
    start_pos = body.get_pos().cpu().numpy().copy()
    start_q = body.get_quat().cpu().numpy().copy()
    _, _, start_yaw_deg = quat_to_euler(start_q)
    start_yaw_rad = np.radians(start_yaw_deg)
    for _ in range(settle_steps):
        rigid_solver.clear_external_force()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_w = body.get_ang().cpu().numpy()
        T_total, q_des = position_pd(start_pos, pos, vel, start_yaw_rad)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)
        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        thrusts = np.clip(thrusts, 0.0, None)
        apply_rotor_forces(rigid_solver, link_idx, thrusts)
        scene.step()
    return sensor.read().distances.flatten()[0].item()


def run_argus_tests():
    print(f"\n{'='*50}")
    print(f"  ARGUS Sensor Tests")
    print(f"{'='*50}")

    passed = True
    emitter_arm = BODY_W / 2

    # Test 1: wall 3.0 m from emitter
    reset_body(pos=(0.0, 0.0, 1.0))
    r1 = read_argus(argus_0deg)
    expected_1 = wall_surface_x - emitter_arm
    ok1 = abs(r1 - expected_1) < 0.05
    passed &= ok1
    print(f"  Test 1 (wall {expected_1:.1f} m): range={r1:.4f}  {'PASS' if ok1 else 'FAIL'}")

    # Test 2: move drone +0.5 m in X
    reset_body(pos=(0.5, 0.0, 1.0))
    r2 = read_argus(argus_0deg)
    expected_2 = expected_1 - 0.5
    ok2 = abs(r2 - expected_2) < 0.05
    passed &= ok2
    print(f"  Test 2 (moved +0.5 m): range={r2:.4f}  {'PASS' if ok2 else 'FAIL'}")

    # Test 3: yaw 20°, beam rotates with body
    q_20 = np.array([np.cos(np.radians(10.0)), 0.0, 0.0, np.sin(np.radians(10.0))])
    reset_body(pos=(0.0, 0.0, 1.0), quat=q_20)
    r3 = read_argus(argus_0deg)
    cos20 = np.cos(np.radians(20.0))
    # ray goes [cos20, sin20, 0]; t such that x = wall_surface_x
    t3 = (wall_surface_x - emitter_arm * cos20) / cos20
    expected_3 = t3
    ok3 = abs(r3 - expected_3) < 0.05
    passed &= ok3
    print(f"  Test 3 (yaw 20°): range={r3:.4f} (expected {expected_3:.4f})  {'PASS' if ok3 else 'FAIL'}")

    # Test 4: pitch gimbal 15°
    reset_body(pos=(0.0, 0.0, 1.0))
    r4 = read_argus(argus_15deg)
    cos15 = np.cos(np.radians(15.0))
    t4 = (wall_surface_x - emitter_arm) / cos15
    expected_4 = t4
    ok4 = abs(r4 - expected_4) < 0.05
    passed &= ok4
    print(f"  Test 4 (pitch 15°): range={r4:.4f} (expected {expected_4:.4f})  {'PASS' if ok4 else 'FAIL'}")

    # Test 5: no intersection (yaw 180°)
    q_180 = np.array([0.0, 0.0, 0.0, 1.0])
    reset_body(pos=(0.0, 0.0, 1.0), quat=q_180)
    r5 = read_argus(argus_0deg)
    ok5 = r5 == -1.0
    passed &= ok5
    print(f"  Test 5 (yaw 180°, no hit): range={r5:.1f}  {'PASS' if ok5 else 'FAIL'}")

    print(f"\n  ARGUS {'PASS' if passed else 'FAIL'}")
    return passed


def run_argus_trajectory_export(path="/workspace/echos_argus_trajectory.npz", steps=120):
    """Record a fresh Genesis trajectory using the actual ARGUS sensor state."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    reset_body(pos=(0.0, 0.0, 1.0))
    recorder = TrajectoryRecorder(physics_dt_s=SIM_DT_S, control_dt_s=SIM_DT_S)
    for step in range(steps):
        use_pitch = step >= steps // 2
        sensor = argus_15deg if use_pitch else argus_0deg
        pitch = np.radians(15.0) if use_pitch else 0.0
        sensor.read()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_world = body.get_ang().cpu().numpy()
        recorder.record(
            step * SIM_DT_S, pos, q_cur, vel, omega_world,
            GimbalState.from_pitch(pitch),
        )
        rigid_solver.clear_external_force()
        T_total, q_des = position_pd(np.array([1.0, 0.0, 1.0]), pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_world)
        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        apply_rotor_forces(rigid_solver, link_idx, np.clip(thrusts, 0.0, None))
        scene.step()
    recorder.save(path)
    print(f"ARGUS_TRAJECTORY_EXPORT_PASS path={path} frames={steps}")
    return path


def run_argus_scan(print_output=True):
    """Deterministic pitch scan: -30° to +30°, 5° steps, 13 beams."""
    reset_body(pos=(0.0, 0.0, 1.0))
    for _ in range(10):
        rigid_solver.clear_external_force()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_w = body.get_ang().cpu().numpy()
        T_total, q_des = position_pd(np.array([0.0, 0.0, 1.0]), pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)
        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        thrusts = np.clip(thrusts, 0.0, None)
        apply_rotor_forces(rigid_solver, link_idx, thrusts)
        scene.step()

    data = argus_scan.read()
    dist = data.distances.cpu().numpy()
    ranges = dist.flatten()

    n_pitch = len(pitch_angles_deg)
    if len(ranges) != n_pitch:
        ranges = ranges[:n_pitch]

    cols = []
    for i, pitch_deg in enumerate(pitch_angles_deg):
        theta = np.radians(pitch_deg)
        rng = ranges[i]
        valid = rng >= 0.0
        if valid:
            pt_body = np.array([rng * np.cos(theta) + emitter_offset[0], 0.0, rng * np.sin(theta)])
        else:
            pt_body = None
        cols.append({'pitch_deg': pitch_deg, 'range': rng, 'valid': valid, 'hit_body': pt_body})

    if print_output:
        print(f"\n{'='*50}")
        print(f"  ARGUS Pitch Scan ({pitch_angles_deg[0]:.0f}° to {pitch_angles_deg[-1]:.0f}°, 5° steps)")
        print(f"{'='*50}")
        print(f"  {'Pitch':>6s}  {'Range':>8s}  {'X_body':>8s}  {'Z_body':>8s}  {'Surface':>8s}")
        print(f"  {'-'*46}")
        for c in cols:
            if c['valid']:
                z = c['hit_body'][2]
                if abs(c['hit_body'][0] - wall_surface_x) < 0.05:
                    surf = "wall"
                elif z < -0.5:
                    surf = "floor"
                else:
                    surf = "?"
                print(f"  {c['pitch_deg']:>5d}°  {c['range']:>8.4f}  {c['hit_body'][0]:>8.4f}  {z:>8.4f}  {surf:>8s}")
            else:
                print(f"  {c['pitch_deg']:>5d}°  {c['range']:>8.4f}  {'---':>8s}  {'---':>8s}  {'miss':>8s}")

    return cols


def run_scan_tests():
    print(f"\n{'='*50}")
    print(f"  ARGUS Scan Tests")
    print(f"{'='*50}")
    passed = True
    emitter_arm = BODY_W / 2

    scan1 = run_argus_scan(print_output=True)

    # --- Test 1: wall plane-intersection geometry ---
    wall_checks = []
    for c in scan1:
        if not c['valid']:
            continue
        theta = np.radians(c['pitch_deg'])
        exp = (wall_surface_x - emitter_arm) / np.cos(theta)
        z_at_wall = 1.0 + exp * np.sin(theta)
        if 0.0 <= z_at_wall <= 2.0:
            wall_checks.append((c['pitch_deg'], c['range'], exp))
    ok1 = all(abs(a[1] / a[2] - 1.0) < 0.02 for a in wall_checks)
    passed &= ok1
    print(f"\n  Test 1 (wall plane-intersection): {'PASS' if ok1 else 'FAIL'}")
    for pd_, r_, e_ in wall_checks:
        print(f"    {pd_:>3d}°: range={r_:.4f}  expected={e_:.4f}  ratio={r_/e_:.4f}")

    # --- Test 2: downward beams hit floor (z_body ≈ -1.0) ---
    floor_hits = []
    for c in scan1:
        if not c['valid']:
            continue
        theta = np.radians(c['pitch_deg'])
        exp_r = (wall_surface_x - emitter_arm) / np.cos(theta)
        z_at_wall = 1.0 + exp_r * np.sin(theta)
        if z_at_wall < 0.0:
            floor_hits.append(c)
    ok2 = True
    for c in floor_hits:
        theta = np.radians(c['pitch_deg'])
        t_floor = -1.0 / np.sin(theta)
        x_exp = emitter_arm + t_floor * np.cos(theta)
        z_err = abs(c['hit_body'][2] + 1.0)
        x_err = abs(c['hit_body'][0] - x_exp)
        ok2 = ok2 and z_err < 0.02 and x_err < 0.05
    if len(floor_hits) < 3:
        ok2 = False
    passed &= ok2
    print(f"\n  Test 2 (floor hits): {'PASS' if ok2 else 'FAIL'} (found {len(floor_hits)} floor rays)")
    for c in floor_hits:
        theta = np.radians(c['pitch_deg'])
        t_floor = -1.0 / np.sin(theta)
        x_exp = emitter_arm + t_floor * np.cos(theta)
        print(f"    {c['pitch_deg']:>3d}°: range={c['range']:.4f}  x={c['hit_body'][0]:.4f} (exp {x_exp:.4f})  z={c['hit_body'][2]:.4f} (exp -1.0)")

    # --- Test 3: upward beams with no surface (go over wall) ---
    misses = [c for c in scan1 if not c['valid']]
    ok3 = all(c['range'] == -1.0 for c in misses) and len(misses) >= 3
    passed &= ok3
    print(f"\n  Test 3 (no returns): {'PASS' if ok3 else 'FAIL'} (found {len(misses)} misses)")
    for c in misses:
        print(f"    {c['pitch_deg']:>3d}°: range={c['range']:.1f}")

    # --- Test 4: repeatability ---
    scan2 = run_argus_scan(print_output=False)
    ok4 = True
    for c1, c2 in zip(scan1, scan2):
        if abs(c1['range'] - c2['range']) > 1e-4:
            ok4 = False
            break
        if c1['valid'] and c2['valid'] and np.linalg.norm(c1['hit_body'] - c2['hit_body']) > 1e-4:
            ok4 = False
            break
    passed &= ok4
    print(f"\n  Test 4 (repeatability): {'PASS' if ok4 else 'FAIL'}")

    # --- Test 5: gimbal returns to 0° after scan ---
    zero_idx = np.where(pitch_angles_deg == 0)[0][0]
    c_zero = scan1[zero_idx]
    expected_wall = wall_surface_x - emitter_arm
    ok5 = c_zero['valid'] and abs(c_zero['range'] - expected_wall) < 0.05
    passed &= ok5
    print(f"\n  Test 5 (gimbal at 0°): range={c_zero['range']:.4f}  expected={expected_wall:.4f}  {'PASS' if ok5 else 'FAIL'}")

    # Reconstructed vertical slice
    print(f"\n  Reconstructed vertical slice (body frame):")
    for c in scan1:
        if c['valid']:
            print(f"    x={c['hit_body'][0]:.4f}  z={c['hit_body'][2]:.4f}  ({c['pitch_deg']:+.0f}°)")
        else:
            print(f"    ----  ----  ({c['pitch_deg']:+.0f}°, no return)")

    print(f"\n  SCAN {'PASS' if passed else 'FAIL'}")
    return passed


def run_motion_scan():
    print(f"\n{'='*50}")
    print(f"  Motion-Compensated ARGUS Scan")
    print(f"{'='*50}")
    print(f"  Flight: (0, 0, 1) → (1, 0, 1)")
    print(f"  Continuous pitch scan (-30° to +30°, 5° steps, 13 beams)")

    reset_body(pos=(0.0, 0.0, 1.0))
    target = np.array([1.0, 0.0, 1.0])
    total_steps = 600
    emitter_arm = BODY_W / 2
    raw_rays = []
    pos_0, q_0 = None, None

    for step in range(total_steps):
        if step % 200 == 0:
            print(f"  step {step}/{total_steps}")
        rigid_solver.clear_external_force()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_w = body.get_ang().cpu().numpy()
        T_total, q_des = position_pd(target, pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)
        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        thrusts = np.clip(thrusts, 0.0, None)
        apply_rotor_forces(rigid_solver, link_idx, thrusts)
        scene.step()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        data = argus_scan.read()
        ranges = data.distances.cpu().numpy().flatten()

        if pos_0 is None:
            pos_0, q_0 = pos.copy(), q_cur.copy()

        for i, pitch_deg in enumerate(pitch_angles_deg):
            if i >= len(ranges):
                break
            rng = float(ranges[i])
            theta = np.radians(pitch_deg)
            valid = rng >= 0.0
            pt_body = np.array([rng * np.cos(theta) + emitter_arm, 0.0, rng * np.sin(theta)]) if valid else None
            raw_rays.append({
                'step': step, 'pitch_deg': pitch_deg, 'range': rng,
                'valid': valid, 'pt_body': pt_body,
                'pos': pos.copy(), 'q': q_cur.copy(),
            })

    R_wb_0 = R_world_from_body(q_0)
    wall_comp, wall_uncomp = [], []
    floor_comp, floor_uncomp = [], []
    wall_steps, wall_pitches = [], []
    n_valid, n_miss = 0, 0

    for ray in raw_rays:
        if not ray['valid']:
            n_miss += 1
            continue
        n_valid += 1
        R_wb = R_world_from_body(ray['q'])
        pt_w = ray['pos'] + R_wb @ ray['pt_body']
        pt_u = pos_0 + R_wb_0 @ ray['pt_body']
        ray['pt_world'] = pt_w
        ray['pt_uncomp'] = pt_u

        if abs(pt_w[0] - wall_surface_x) < 0.1:
            wall_comp.append(pt_w)
            wall_uncomp.append(pt_u)
            wall_steps.append(ray['step'])
            wall_pitches.append(ray['pitch_deg'])
        elif abs(pt_w[2]) < 0.1:
            floor_comp.append(pt_w)
            floor_uncomp.append(pt_u)

    wall_comp = np.array(wall_comp)
    wall_uncomp = np.array(wall_uncomp)
    floor_comp = np.array(floor_comp)
    floor_uncomp = np.array(floor_uncomp)

    wall_comp_err = wall_comp[:, 0] - wall_surface_x if len(wall_comp) > 0 else np.array([])
    wall_uncomp_err = wall_uncomp[:, 0] - wall_surface_x if len(wall_uncomp) > 0 else np.array([])
    wall_comp_rmse = np.sqrt(np.mean(wall_comp_err**2)) if len(wall_comp_err) > 0 else float('nan')
    wall_uncomp_rmse = np.sqrt(np.mean(wall_uncomp_err**2)) if len(wall_uncomp_err) > 0 else float('nan')
    wall_comp_mean = np.mean(wall_comp_err) if len(wall_comp_err) > 0 else float('nan')
    wall_uncomp_mean = np.mean(wall_uncomp_err) if len(wall_uncomp_err) > 0 else float('nan')

    floor_comp_err = floor_comp[:, 2] if len(floor_comp) > 0 else np.array([])
    floor_uncomp_err = floor_uncomp[:, 2] if len(floor_uncomp) > 0 else np.array([])
    floor_comp_rmse = np.sqrt(np.mean(floor_comp_err**2)) if len(floor_comp_err) > 0 else float('nan')
    floor_uncomp_rmse = np.sqrt(np.mean(floor_uncomp_err**2)) if len(floor_uncomp_err) > 0 else float('nan')
    floor_comp_mean = np.mean(floor_comp_err) if len(floor_comp_err) > 0 else float('nan')
    floor_uncomp_mean = np.mean(floor_uncomp_err) if len(floor_uncomp_err) > 0 else float('nan')

    print(f"\n  Collected: {n_valid} valid rays, {n_miss} misses")
    print(f"  Wall hits: {len(wall_comp)}  Floor hits: {len(floor_comp)}")
    print(f"\n  {'='*52}")
    print(f"  {'Method':>14s}  {'Surface':>8s}  {'Mean Err':>9s}  {'RMSE':>9s}  {'Std':>9s}")
    print(f"  {'-'*52}")
    if len(wall_comp) > 0:
        print(f"  {'Compensated':>14s}  {'wall':>8s}  {wall_comp_mean:>+9.4f}  {wall_comp_rmse:>9.4f}  {np.std(wall_comp_err):>9.4f}")
        print(f"  {'Uncompensated':>14s}  {'wall':>8s}  {wall_uncomp_mean:>+9.4f}  {wall_uncomp_rmse:>9.4f}  {np.std(wall_uncomp_err):>9.4f}")
    if len(floor_comp) > 0:
        print(f"  {'Compensated':>14s}  {'floor':>8s}  {floor_comp_mean:>+9.4f}  {floor_comp_rmse:>9.4f}  {np.std(floor_comp_err):>9.4f}")
        print(f"  {'Uncompensated':>14s}  {'floor':>8s}  {floor_uncomp_mean:>+9.4f}  {floor_uncomp_rmse:>9.4f}  {np.std(floor_uncomp_err):>9.4f}")

    print(f"\n  Sample wall points (compensated → uncompensated):")
    for i in range(min(5, len(wall_comp))):
        c = wall_comp[i]
        u = wall_uncomp[i]
        print(f"    step {wall_steps[i]:3d}  {wall_pitches[i]:+3d}°: ({c[0]:.4f},{c[1]:.4f},{c[2]:.4f}) → ({u[0]:.4f},{u[1]:.4f},{u[2]:.4f})")

    print(f"\n  {'='*50}")
    print(f"  Motion Scan Tests")
    print(f"  {'='*50}")
    passed = True

    ok1 = len(wall_comp) > 0 and wall_comp_rmse < 0.05
    passed &= ok1
    print(f"\n  Test 1 (wall plane fixity): rmse={wall_comp_rmse:.4f} m  {'PASS' if ok1 else 'FAIL'}")

    ok2 = len(wall_comp) > 0 and abs(wall_comp_mean) < 0.01
    passed &= ok2
    print(f"  Test 2 (plane pos error < 1 cm): mean={abs(wall_comp_mean):.4f} m  {'PASS' if ok2 else 'FAIL'}")

    ok3 = len(floor_comp) > 0 and abs(floor_comp_mean) < 0.01
    passed &= ok3
    print(f"  Test 3 (floor height): mean_z={abs(floor_comp_mean):.4f} m  {'PASS' if ok3 else 'FAIL'}")

    ok4 = len(wall_comp) > 0 and len(wall_uncomp) > 0 and wall_uncomp_rmse > 10 * wall_comp_rmse
    passed &= ok4
    print(f"  Test 4 (uncompensated fails): uncomp_rmse={wall_uncomp_rmse:.4f} vs comp_rmse={wall_comp_rmse:.4f}  {'PASS' if ok4 else 'FAIL'}")

    ok5 = all(r['range'] == -1.0 for r in raw_rays if not r['valid'])
    passed &= ok5
    print(f"  Test 5 (no-return excluded): {n_miss} misses at range=-1.0  {'PASS' if ok5 else 'FAIL'}")

    print(f"\n  MOTION SCAN {'PASS' if passed else 'FAIL'}")
    return passed


def run_3d_stationary_scan():
    print(f"\n{'='*50}")
    print(f"  3D Stationary ARGUS Scan")
    print(f"{'='*50}")
    print(f"  Yaw: -60° to +60° (10° steps)")
    print(f"  Pitch: -30° to +30° (5° steps)")
    print(f"  Scene: floor + two perpendicular walls (corner at x={wall_surface_x}, y={wall_surface_y})")

    yaw_angles_deg = np.arange(-60, 61, 10)
    n_yaw = len(yaw_angles_deg)
    emitter_arm = BODY_W / 2
    settle_steps = 20
    ply_path = "/workspace/argus_3d_scan.ply"

    def acquire_scan(label=""):
        all_rays = []
        for yi, yaw_deg in enumerate(yaw_angles_deg):
            psi_des = np.radians(yaw_deg)
            _, q_des = position_pd(np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), psi_des)
            reset_body(pos=(0.0, 0.0, 1.0), quat=q_des)
            for _ in range(settle_steps):
                rigid_solver.clear_external_force()
                qc = body.get_quat().cpu().numpy()
                p = body.get_pos().cpu().numpy()
                v = body.get_vel().cpu().numpy()
                om = body.get_ang().cpu().numpy()
                Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v, psi_des)
                ta, _ = attitude_pd(qd, qc, om)
                th, _, _ = mixer(Tt, ta[0], ta[1], ta[2])
                apply_rotor_forces(rigid_solver, link_idx, np.clip(th, 0.0, None))
                scene.step()
            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            data = argus_scan.read()
            ranges = data.distances.cpu().numpy().flatten()
            if label:
                print(f"  {label} yaw {yaw_deg:+3d}° ({yi+1}/{n_yaw})  ", end="")
            for i, pitch_deg in enumerate(pitch_angles_deg):
                rng = float(ranges[i]) if i < len(ranges) else -1.0
                theta = np.radians(pitch_deg)
                valid = rng >= 0.0
                pb = np.array([rng * np.cos(theta) + emitter_arm, 0.0, rng * np.sin(theta)]) if valid else None
                all_rays.append({
                    'yaw_deg': yaw_deg, 'pitch_deg': pitch_deg,
                    'range': rng, 'valid': valid, 'pt_body': pb,
                    'pos': pos.copy(), 'q': q_cur.copy(),
                })
            if label and yi % 3 == 0:
                print(f"  {label} yaw {yaw_deg:+3d}° ({yi+1}/{n_yaw})")
        return all_rays

    print(f"  Acquiring scan 1/2...")
    rays1 = acquire_scan("  [1/2]")
    print(f"  Acquiring scan 2/2...")
    rays2 = acquire_scan("  [2/2]")

    def classify_rays(rays):
        w1, w2, fl, ot = [], [], [], []
        w2_yspan = (-4.0, 4.0)
        for ray in rays:
            R_wb = R_world_from_body(ray['q'])
            pos = ray['pos']
            theta = np.radians(ray['pitch_deg'])
            dir_body = np.array([np.cos(theta), 0.0, np.sin(theta)])
            dir_world = R_wb @ dir_body
            emitter_w = pos + R_wb @ np.array([emitter_offset[0], 0.0, 0.0])
            if ray['valid']:
                pt = ray['pos'] + R_wb @ ray['pt_body']
                ray['pt_world'] = pt
                on_xwall = abs(pt[0] - wall_surface_x) < 0.1 and abs(pt[2]) > 0.05
                on_ywall = abs(pt[1] - wall_surface_y) < 0.1 and abs(pt[2]) > 0.05
                on_floor = abs(pt[2]) < 0.1
                if on_xwall:
                    w1.append(pt)
                elif on_ywall:
                    w2.append(pt)
                elif on_floor:
                    fl.append(pt)
                else:
                    ot.append(pt)
            else:
                ot.append(None)
        return np.array(w1), np.array(w2), np.array(fl), np.array(ot)

    w1_1, w2_1, fl_1, ot_1 = classify_rays(rays1)
    w1_2, w2_2, fl_2, ot_2 = classify_rays(rays2)

    def fit_plane(pts):
        c = np.mean(pts, axis=0)
        _, _, Vt = np.linalg.svd(pts - c)
        n = Vt[-1]
        if n @ c < 0:
            n = -n
        d = -n @ c
        errs = pts @ n + d
        rmse = np.sqrt(np.mean(errs ** 2))
        return n, d, c, rmse

    w1_n, w1_d, w1_c, w1_rmse = fit_plane(w1_1) if len(w1_1) >= 3 else (np.zeros(3), 0, np.zeros(3), float('nan'))
    w2_n, w2_d, w2_c, w2_rmse = fit_plane(w2_1) if len(w2_1) >= 3 else (np.zeros(3), 0, np.zeros(3), float('nan'))
    fl_n, fl_d, fl_c, fl_rmse = fit_plane(fl_1) if len(fl_1) >= 3 else (np.zeros(3), 0, np.zeros(3), float('nan'))
    wall_angle = np.degrees(np.arccos(np.clip(np.abs(w1_n @ w2_n), 0, 1))) if len(w1_1) >= 3 and len(w2_1) >= 3 else float('nan')

    # Write PLY
    all_pts = [w1_1, w2_1, fl_1]
    n_total = sum(len(p) for p in all_pts)
    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n_total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for pts in all_pts:
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    # Print results
    n_valid1 = sum(1 for r in rays1 if r['valid'])
    n_miss1 = sum(1 for r in rays1 if not r['valid'])
    print(f"\n  Scan totals: {n_valid1} valid, {n_miss1} misses")
    print(f"    Wall X (x≈{wall_surface_x}): {len(w1_1)} pts")
    print(f"    Wall Y (y≈{wall_surface_y}): {len(w2_1)} pts")
    print(f"    Floor: {len(fl_1)} pts")
    print(f"    Other: {len(ot_1)} pts")
    print(f"    PLY saved: {ply_path}")

    def planestr(n, d):
        return f"{n[0]:.4f}x + {n[1]:.4f}y + {n[2]:.4f}z + {d:.4f} = 0"

    print(f"\n  Fitted planes:")
    if len(w1_1) >= 3:
        print(f"    Wall X: {planestr(w1_n, w1_d)}  (rmse={w1_rmse:.4f})")
    if len(w2_1) >= 3:
        print(f"    Wall Y: {planestr(w2_n, w2_d)}  (rmse={w2_rmse:.4f})")
    if len(fl_1) >= 3:
        print(f"    Floor:  {planestr(fl_n, fl_d)}  (rmse={fl_rmse:.4f})")
    if not np.isnan(wall_angle):
        print(f"    Wall angle: {wall_angle:.2f}°")

    # NumPy output
    dtype = [('x', float), ('y', float), ('z', float), ('surface', 'U5')]
    npy_rows = []
    for pts, surf in [(w1_1, 'wall_x'), (w2_1, 'wall_y'), (fl_1, 'floor')]:
        for p in pts:
            npy_rows.append((p[0], p[1], p[2], surf))
    point_cloud = np.array(npy_rows, dtype=dtype)
    np.savez_compressed("/workspace/argus_3d_scan.npz", cloud=point_cloud)
    print(f"    Numpy cloud saved: /workspace/argus_3d_scan.npz  ({len(point_cloud)} pts)")

    print(f"\n  Sample points (10 each):")
    for pts, label in [(w1_1, 'wall_x'), (w2_1, 'wall_y'), (fl_1, 'floor')]:
        print(f"    {label} ({len(pts)} pts):")
        for i in range(min(10, len(pts))):
            print(f"      ({pts[i][0]:.4f}, {pts[i][1]:.4f}, {pts[i][2]:.4f})")

    # --- Tests ---
    print(f"\n{'='*50}")
    print(f"  3D Scan Tests")
    print(f"{'='*50}")
    passed = True

    # Test 1: Both wall planes receive ray hits
    ok1 = len(w1_1) > 10 and len(w2_1) > 10
    passed &= ok1
    print(f"\n  Test 1 (two wall planes): Wall X={len(w1_1)} pts, Wall Y={len(w2_1)} pts  {'PASS' if ok1 else 'FAIL'}")

    # Test 2: Each plane RMSE < 1 cm
    ok2x = len(w1_1) >= 3 and w1_rmse < 0.01
    ok2y = len(w2_1) >= 3 and w2_rmse < 0.01
    ok2 = ok2x and ok2y
    passed &= ok2
    print(f"  Test 2 (rmse < 1 cm): x_rmse={w1_rmse:.4f}, y_rmse={w2_rmse:.4f}  {'PASS' if ok2 else 'FAIL'}")

    # Test 3: Wall angle = 90° ± 0.5°
    ok3 = not np.isnan(wall_angle) and abs(wall_angle - 90.0) < 0.5
    passed &= ok3
    print(f"  Test 3 (wall angle 90°): measured={wall_angle:.2f}°  {'PASS' if ok3 else 'FAIL'}")

    # Test 4: Floor points at correct height
    fl_mean_z = np.mean(fl_1[:, 2]) if len(fl_1) > 0 else float('nan')
    ok4 = len(fl_1) > 0 and abs(fl_mean_z) < 0.01
    passed &= ok4
    print(f"  Test 4 (floor height): mean_z={fl_mean_z:.4f}  {'PASS' if ok4 else 'FAIL'}")

    # Test 5: No-return rays excluded
    ok5 = all(r['range'] == -1.0 for r in rays1 if not r['valid']) and n_miss1 > 0
    passed &= ok5
    print(f"  Test 5 (no-return excluded): {n_miss1} misses  {'PASS' if ok5 else 'FAIL'}")

    # Test 6: Repeatability
    ok6 = True
    for r1, r2 in zip(rays1, rays2):
        if abs(r1['range'] - r2['range']) > 1e-4:
            ok6 = False
            break
    passed &= ok6
    print(f"  Test 6 (deterministic): {len(rays1)} rays match  {'PASS' if ok6 else 'FAIL'}")

    # Test 7: Yaw and gimbal return to 0° after scan
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    reset_body(pos=(0.0, 0.0, 1.0), quat=q_id)
    for _ in range(20):
        rigid_solver.clear_external_force()
        qc = body.get_quat().cpu().numpy()
        p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy()
        om = body.get_ang().cpu().numpy()
        Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
        ta, _ = attitude_pd(qd, qc, om)
        th, _, _ = mixer(Tt, ta[0], ta[1], ta[2])
        apply_rotor_forces(rigid_solver, link_idx, np.clip(th, 0.0, None))
        scene.step()
    data_post = argus_scan.read()
    r_post = data_post.distances.cpu().numpy().flatten()
    zero_idx = np.where(pitch_angles_deg == 0)[0][0]
    r0_post = float(r_post[zero_idx]) if zero_idx < len(r_post) else -1.0
    expected_wall = wall_surface_x - emitter_arm
    final_q = body.get_quat().cpu().numpy()
    final_yaw = quat_to_euler(final_q)[2]
    ok7 = abs(final_yaw) < 0.5 and abs(r0_post - expected_wall) < 0.05
    passed &= ok7
    print(f"  Test 7 (return to 0°): yaw={final_yaw:.2f}°, range_0°={r0_post:.4f} (exp {expected_wall:.4f})  {'PASS' if ok7 else 'FAIL'}")

    print(f"\n  3D SCAN {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    import sys
    if "--export-trajectory" in sys.argv:
        index = sys.argv.index("--export-trajectory")
        path = sys.argv[index + 1] if index + 1 < len(sys.argv) else "/workspace/echos_argus_trajectory.npz"
        run_argus_trajectory_export(path)
        raise SystemExit(0)
    results = []
    results.append(run_pos_test("X offset 0.5m", init_pos=(0.5, 0.0, 1.0), pos_des=(0.0, 0.0, 1.0)))
    results.append(run_pos_test("Y offset 0.5m", init_pos=(0.0, 0.5, 1.0), pos_des=(0.0, 0.0, 1.0)))
    results.append(run_pos_test("Z offset 0.5m", init_pos=(0.0, 0.0, 1.5), pos_des=(0.0, 0.0, 1.0)))
    route = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]),
             np.array([1.0, 1.0, 1.0]), np.array([0.0, 1.0, 1.0]),
             np.array([0.0, 0.0, 1.0])]
    results.append(run_waypoint_mission(route))
    results.append(run_hover_gust_test())
    results.append(run_route_gust_test(route))
    results.append(run_yaw_error_test())
    results.append(run_argus_tests())
    results.append(run_scan_tests())
    results.append(run_motion_scan())
    results.append(run_3d_stationary_scan())
    raise SystemExit(0 if all(results) else 1)
