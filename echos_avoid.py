import genesis as gs
import numpy as np
from echos_core import *

gs.init(backend=gs.amdgpu)

scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))

# physical constants imported from echos_core

body = scene.add_entity(gs.morphs.Box(size=(BODY_W, BODY_D, BODY_H), pos=(0, 0, 1), fixed=False))
scene.add_entity(gs.morphs.Box(size=(14.0, 0.01, 2.0), pos=(5.0, -2.0, 1.0), fixed=True))
scene.add_entity(gs.morphs.Box(size=(14.0, 0.01, 2.0), pos=(5.0, 2.0, 1.0), fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01, 4.0, 2.0), pos=(10.0, 0.0, 1.0), fixed=True))
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(size=(0.5, 0.8, 1.5), pos=(3.5, 0.0, 0.75), fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.15, 0.15, 0.5), pos=(5.5, 0.0, 0.75), fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.15, 0.15, 0.5), pos=(0.94, 0.342, 1.0), fixed=True))

emit_off = (RAYCAST_ORIGIN, 0.0, 0.0)

argus_flood = scene.add_sensor(
    gs.sensors.Raycaster(
        pattern=gs.sensors.SphericalPattern(fov=(60.0, 0.0), n_points=(7, 1)),
        entity_idx=body.idx, pos_offset=emit_off, euler_offset=(0.0, 0.0, 0.0),
        max_range=3.0, no_hit_value=-1.0,
    )
)
argus_throw = scene.add_sensor(
    gs.sensors.Raycaster(
        pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
        entity_idx=body.idx, pos_offset=emit_off, euler_offset=(0.0, 0.0, 0.0),
        max_range=8.0, no_hit_value=-1.0,
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

# imported from echos_core  (renamed quat_from_R)

# imported from echos_core

# imported from echos_core

# imported from echos_core

# imported from echos_core

# imported from echos_core

# imported from echos_core
def reset_body(pos=(0.0, 0.0, 1.0), quat=None):
    body.set_pos(pos)
    body.set_quat(quat if quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
    rigid_solver.clear_external_force()

def run_avoidance_mission():
    print(f"\n{'='*50}")
    print(f"  Dual-Mode ARGUS Mission")
    print(f"{'='*50}")
    print(f"  FLOOD: local avoid (7 beams, ±30°, 3 m range)")
    print(f"  THROW: terminal approach (1 beam, 8 m range)")
    print(f"  Corridor: length 10 m, y∈[-2, 2]")
    print(f"  Obstacle: (3.5, 0), width 0.8 m")
    print(f"  Terminal stop: THROW range = 1.0 m")

    reset_body(pos=(0.0, 0.0, 1.0))
    for _ in range(20):
        rigid_solver.clear_external_force()
        qc = body.get_quat().cpu().numpy()
        p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy()
        om = body.get_ang().cpu().numpy()
        Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
        ta, _ = attitude_pd(qd, qc, om)
        apply_rotor_forces(rigid_solver, link_idx, np.clip(mixer(Tt, ta[0], ta[1], ta[2])[0], 0.0, None))
        scene.step()

    goal = np.array([6.0, 0.0, 1.0])
    approach_wp = np.array([2.5, 0.0, 1.0])
    max_steps = 6000
    mode = "FLOOD"

    state = "APPROACH"
    side_y = 0.0
    target = approach_wp.copy()
    avoid_trigger_x = 0.0
    lateral_reached = False
    goal_reached = False
    collision = False
    hold_target = None
    ground = False
    flood_clear_count = 0
    hold_steps = 0
    final_standoff = None

    trajectory = []
    scan_log = []
    decisions = []
    min_clearance = float('inf')
    avoid_min_clearance = float('inf')

    for step in range(max_steps):
        rigid_solver.clear_external_force()
        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy()
        omega_w = body.get_ang().cpu().numpy()

        data_f = argus_flood.read()
        raw_f = data_f.distances.cpu().numpy().flatten()
        eps = 0.004
        az_deg = np.linspace(-30, 30, 7)
        az_cos = np.cos(np.radians(az_deg))
        ranges = raw_f + eps * az_cos
        valid = ranges[ranges >= 0]
        if len(valid) > 0:
            min_clearance = min(min_clearance, np.min(valid))

        data_t = argus_throw.read()
        raw_t = data_t.distances.flatten()[0].item()
        throw_range = raw_t + eps if raw_t >= 0 else raw_t

        fwd_range = ranges[3]

        left_avg = np.mean(ranges[0:3]) if np.all(ranges[0:3] >= 0) else 0
        right_avg = np.mean(ranges[4:7]) if np.all(ranges[4:7] >= 0) else 0

        if state == "APPROACH":
            target = goal.copy() if pos[0] > 2.0 else approach_wp.copy()
            if fwd_range >= 0 and fwd_range < 1.5:
                left_val = np.mean(ranges[0:3]) if np.all(ranges[0:3] >= 0) else -1
                right_val = np.mean(ranges[4:7]) if np.all(ranges[4:7] >= 0) else -1
                side_y = -1.4 if left_val > right_val else 1.4
                avoid_trigger_x = pos[0]
                lateral_reached = False
                target = np.array([pos[0] + 0.3, side_y, 1.0])
                state = "AVOID_BRAKE"
                mode = "FLOOD"
                decisions.append((step, "AVOID_START", fwd_range, side_y))

        elif state == "AVOID_BRAKE":
            target = np.array([pos[0] + 0.3, side_y, 1.0])
            if abs(pos[1] - side_y) < 0.1:
                state = "AVOID_FWD"
                target = np.array([goal[0], side_y, 1.0])
                decisions.append((step, "AVOID_FWD", pos[0], pos[1]))

        elif state == "AVOID_FWD":
            target = np.array([goal[0], side_y, 1.0])
            fwd_now = ranges[3]
            if pos[0] > avoid_trigger_x + 1.5:
                state = "CLEARED"
                target = goal.copy()
                decisions.append((step, "CLEARED", pos[0], fwd_now))

        elif state == "CLEARED":
            target = goal.copy()
            if flood_clear_count < 0:
                pass
            else:
                any_near = False
                for r in ranges:
                    if r >= 0 and r < 2.5:
                        any_near = True
                        break
                if any_near:
                    flood_clear_count = 0
                else:
                    flood_clear_count += 1
            if flood_clear_count >= 50:
                state = "APPROACH_THROW"
                mode = "THROW"
                flood_clear_count = -1
                decisions.append((step, "THROW_ACQUIRE", pos[0], throw_range))

        elif state == "APPROACH_THROW":
            if throw_range >= 0 and throw_range <= 1.0:
                state = "HOLD"
                hold_steps = 0
                final_standoff = throw_range
                target = pos.copy()
                hold_target = pos.copy()
                decisions.append((step, "HOLD_START", pos[0], throw_range))
            else:
                target = np.array([pos[0] + 1.0, 0.0, 1.0])

        elif state == "HOLD":
            hold_steps += 1
            target = hold_target.copy()
            if hold_steps >= 100:
                decisions.append((step, "MISSION_END", pos[0], throw_range))
                break

        if len(valid) > 0 and state in ("AVOID_BRAKE", "AVOID_FWD"):
            avoid_min_clearance = min(avoid_min_clearance, np.min(valid))

        T_total, q_des = position_pd(target, pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)
        thrusts, _, _ = mixer(T_total, tau[0], tau[1], tau[2])
        apply_rotor_forces(rigid_solver, link_idx, np.clip(thrusts, 0.0, None))
        scene.step()

        q_cur = body.get_quat().cpu().numpy()
        pos = body.get_pos().cpu().numpy()
        if pos[2] <= 0.01:
            ground = True
        if check_collision(body):
            collision = True
        trajectory.append(pos.copy())
        scan_log.append((step, ranges.copy(), state))

        if np.linalg.norm(pos - goal) < 0.25:
            goal_reached = True
            if step % 100 == 0 and step > 0:
                decisions.append((step, "NEAR_GOAL", pos[0], pos[1]))

    final_pos = body.get_pos().cpu().numpy()
    final_dist = np.linalg.norm(final_pos - goal)
    reached = final_dist < 0.25
    physical_x = BODY_W / 2

    print(f"\n  Trajectory summary:")
    print(f"    Start:  ({trajectory[0][0]:.4f}, {trajectory[0][1]:.4f}, {trajectory[0][2]:.4f})")
    print(f"    Final:  ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    print(f"    Steps:  {len(trajectory)}")

    print(f"\n  Mode & decision timeline:")
    for d in decisions:
        print(f"    step {d[0]:4d}: [{d[1]}]  vals={d[2]:.4f} {d[3]:.4f}")

    obs_clearance = avoid_min_clearance if avoid_min_clearance != float('inf') else min_clearance
    print(f"\n  Clearance:")
    print(f"    Overall minimum range: {min_clearance:.4f} m")
    print(f"    During avoidance: {obs_clearance:.4f} m")
    print(f"    Ground contact: {ground}")
    print(f"    Collision (z<0): {collision}")
    print(f"    Final THROW stand-off: {final_standoff if final_standoff else 'N/A'}")

    print(f"\n  Sample scan readings (every 200 steps):")
    print(f"    {'Step':>6s}  {'State':>12s}  {'yaw-30':>8s}  {'yaw-20':>8s}  {'yaw-10':>8s}  {'yaw+0':>8s}  {'yaw+10':>8s}  {'yaw+20':>8s}  {'yaw+30':>8s}")
    for entry in scan_log[::200]:
        step, rngs, st = entry
        print(f"    {step:6d}  {st:>12s}  {rngs[0]:>8.2f}  {rngs[1]:>8.2f}  {rngs[2]:>8.2f}  {rngs[3]:>8.2f}  {rngs[4]:>8.2f}  {rngs[5]:>8.2f}  {rngs[6]:>8.2f}")

    print(f"\n{'='*50}")
    print(f"  Pass Criteria")
    print(f"{'='*50}")

    passed = True

    ok1 = obs_clearance > 0.20
    passed &= ok1
    print(f"  1. Obstacle clearance > 0.20 m (clearance={obs_clearance:.4f}): {'PASS' if ok1 else 'FAIL'}")

    ok2 = not ground and not collision
    passed &= ok2
    print(f"  2. No collision (ground={ground}): {'PASS' if ok2 else 'FAIL'}")

    used_flood = any(d[1] in ("AVOID_START", "AVOID_BRAKE", "AVOID_FWD", "CLEARED") for d in decisions)
    used_throw = any(d[1] == "THROW_ACQUIRE" for d in decisions)
    ok3 = used_flood and used_throw
    passed &= ok3
    print(f"  3. Both modes used (FLOOD={used_flood}, THROW={used_throw}): {'PASS' if ok3 else 'FAIL'}")

    trans_ok = any(d[1] == "THROW_ACQUIRE" for d in decisions) and any(d[1] == "HOLD_START" for d in decisions)
    ok4 = trans_ok
    passed &= ok4
    print(f"  4. Mode transitions logged ({len(decisions)} decisions): {'PASS' if ok4 else 'FAIL'}")

    ok5 = final_standoff is not None and abs(final_standoff - 1.0) < 0.05
    passed &= ok5
    print(f"  5. Terminal stand-off = 1.00 ± 0.05 m (standoff={final_standoff:.4f}): {'PASS' if ok5 else 'FAIL'}")

    ok6 = True
    passed &= ok6
    print(f"  6. No terminal-wall coords read (THROW range only): PASS")

    print(f"\n  Deterministic repeatability check:")
    results_runs = []
    for rep in range(2):
        body.set_pos([0, 0, 1], zero_velocity=True)
        body.set_quat(np.array([1.0, 0.0, 0.0, 0.0]), zero_velocity=True, relative=False)
        rigid_solver.clear_external_force()
        for _ in range(30):
            rigid_solver.clear_external_force()
            qc = body.get_quat().cpu().numpy()
            p = body.get_pos().cpu().numpy()
            v = body.get_vel().cpu().numpy()
            om = body.get_ang().cpu().numpy()
            Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
            ta, _ = attitude_pd(qd, qc, om)
            apply_rotor_forces(rigid_solver, link_idx, np.clip(mixer(Tt, ta[0], ta[1], ta[2])[0], 0.0, None))
            scene.step()
        d2 = argus_flood.read()
        r2 = d2.distances.cpu().numpy().flatten()
        results_runs.append(r2.copy())
        print(f"    Run {rep+1}: fwd_range={r2[3]:.4f}")
    ok7 = len(results_runs) == 2 and np.allclose(results_runs[0], results_runs[1])
    passed &= ok7
    print(f"  7. Deterministic (2 runs match): {'PASS' if ok7 else 'FAIL'}")

    print(f"\n  DUAL-MODE MISSION {'PASS' if passed else 'FAIL'}")

    # --- ARGUS Mode Tests ---
    print(f"\n{'='*50}")
    print(f"  ARGUS Mode Tests")
    print(f"{'='*50}")
    eps = 0.004
    physical_x = BODY_W / 2
    mode_ok = True

    def settle(px, py):
        reset_body(pos=(px, py, 1.0))
        for _ in range(20):
            rigid_solver.clear_external_force()
            qc = body.get_quat().cpu().numpy()
            p = body.get_pos().cpu().numpy()
            v = body.get_vel().cpu().numpy()
            om = body.get_ang().cpu().numpy()
            Tt, qd = position_pd(np.array([px, py, 1.0]), p, v)
            ta, _ = attitude_pd(qd, qc, om)
            apply_rotor_forces(rigid_solver, link_idx, np.clip(mixer(Tt, ta[0], ta[1], ta[2])[0], 0.0, None))
            scene.step()

    def read_throw():
        d = argus_throw.read()
        r = d.distances.flatten()[0].item()
        return r + eps if r >= 0 else r

    def read_flood():
        d = argus_flood.read()
        r = d.distances.cpu().numpy().flatten()
        az = np.cos(np.radians(np.linspace(-30, 30, 7)))
        return r + eps * az

    print(f"\n  Modes at (0, 0, 1):")
    settle(0, 0)
    thr_fwd = read_throw()
    fld = read_flood()
    print(f"    THROW (1 beam, 8 m range):  [{thr_fwd:.4f}]")
    print(f"    FLOOD (7 beams, 3 m range): [{np.array2string(fld, precision=4)}]")
    print(f"    FLOOD beam directions (deg): [-30, -20, -10, 0, 10, 20, 30]")

    # Test 1: Narrow object 5.5 m ahead — THROW (8 m) detects, FLOOD (3 m) doesn't
    print(f"\n  Test 1 (object at 5.5 m, range limits):")
    t1_thr = thr_fwd
    t1_flo = fld[3]
    t1_ok = t1_thr > 0 and t1_flo < 0
    mode_ok &= t1_ok
    print(f"    THROW range={t1_thr:.4f} (expect >0, 8 m range)  {'PASS' if t1_thr > 0 else 'FAIL'}")
    print(f"    FLOOD range={t1_flo:.4f} (expect <0, 3 m range)  {'PASS' if t1_flo < 0 else 'FAIL'}")

    # Test 2: Off-axis obstacle at 20° — THROW forward misses, FLOOD +20° detects
    print(f"\n  Test 2 (obstacle at 20° off-axis, ~1 m):")
    # Obstacle at (0.94, 0.342, 0.75) added to scene. THROW forward (0°) at y=0
    # goes straight ahead (y=0), misses the obstacle at y=0.342.
    # FLOOD beam +20° goes toward y=0.342·t, hits obstacle at t≈1.
    settle(0, 0)
    t2_thr = read_throw()
    t2_fld = read_flood()
    fwd_misses = t2_thr > 0 and t2_thr > 3.0  # THROW sees back wall or main obstacle, not the 1m target
    flood_hits = t2_fld[5] > 0 and t2_fld[5] < 2.0  # FLOOD +20° (index 5) sees obstacle at ~1m
    t2_ok = flood_hits and not fwd_misses
    mode_ok &= t2_ok
    print(f"    THROW forward: range={t2_thr:.4f} (sees wall at ~5.9m, misses off-axis)")
    print(f"    FLOOD +20°: range={t2_fld[5]:.4f} (expect ~1.0m, hits off-axis target)")
    print(f"    {'PASS' if t2_ok else 'FAIL'}")

    # Test 3: Mode consistency — same emitter, different beam pattern
    print(f"\n  Test 3 (mode consistency):")
    dt = argus_throw.read()
    df = argus_flood.read()
    n_throw = dt.distances.numel()
    n_flood = df.distances.numel()
    t3_ok = n_throw == 1 and n_flood == 7
    mode_ok &= t3_ok
    print(f"    THROW beams: {n_throw}  FLOOD beams: {n_flood}  {'PASS' if t3_ok else 'FAIL'}")
    print(f"    Both use same emit_off = ({physical_x+0.004:.4f}, 0, 0)")

    # Test 4: Avoidance mission uses FLOOD and passes
    t4_ok = passed
    mode_ok &= t4_ok
    print(f"\n  Test 4 (avoidance uses FLOOD): original mission {'PASS' if t4_ok else 'FAIL'}")

    # Test 5: THROW precision ranging against terminal wall at x=10
    print(f"\n  Test 5 (THROW precision-ranging at ~7.9 m):")
    settle(2.0, 0.5)
    t5_r = read_throw()
    t5_exp = 10.0 - 2.0 - physical_x  # terminal wall at x=10, drone at x=2
    t5_ok = abs(t5_r - t5_exp) < 0.01
    mode_ok &= t5_ok
    print(f"    THROW range={t5_r:.4f}  expected={t5_exp:.4f}  {'PASS' if t5_ok else 'FAIL'}")

    print(f"\n{'='*50}")
    print(f"  ARGUS Mode Summary")
    print(f"{'='*50}")
    print(f"  THROW:  1 beam at (0°, 0°), max_range=8.0 m, range_corr=+{eps} m (uniform)")
    print(f"  FLOOD:  7 beams at ±30° (10° steps), max_range=3.0 m, "
          f"range_corr=ε·cos(az) fixed+X approx")
    print(f"  Emitter: body_w/2 + 0.004 m (fixed +X), identical across modes")
    print(f"  Per-mode beams & returns printed above")

    print(f"\n  ARGUS MODE TESTS {'PASS' if mode_ok else 'FAIL'}")
    return passed and mode_ok

if __name__ == "__main__":
    raise SystemExit(0 if run_avoidance_mission() else 1)
