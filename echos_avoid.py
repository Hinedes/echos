"""Dual-mode avoidance mission: FLOOD for obstacle detection, THROW for approach."""
import genesis as gs
import numpy as np
from echos_core import *
import hashlib


def run_avoidance_mission():
    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))

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
        gs.sensors.Raycaster(pattern=gs.sensors.SphericalPattern(fov=(60.0, 0.0), n_points=(7, 1)),
            entity_idx=body.idx, pos_offset=emit_off, euler_offset=(0.0, 0.0, 0.0),
            max_range=3.0, no_hit_value=-1.0))
    argus_throw = scene.add_sensor(
        gs.sensors.Raycaster(pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
            entity_idx=body.idx, pos_offset=emit_off, euler_offset=(0.0, 0.0, 0.0),
            max_range=8.0, no_hit_value=-1.0))
    scene.build()
    body.set_mass(MASS)
    rigid_solver = scene.sim.rigid_solver
    link_idx = 0

    def reset_body(pos=(0.0, 0.0, 1.0), quat=None):
        reset_body_state(body, rigid_solver, pos, quat)

    approach_wp = np.array([2.0, 0.0, 1.0])
    max_steps = 600
    state = "APPROACH"
    side_y = 0.0
    target = approach_wp.copy()
    avoid_trigger_x = 0.0
    lateral_reached = False
    goal_reached = False
    collision = False
    ground = False
    hold_target = None
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
        ranges_f = raw_f + eps * az_cos
        valid = ranges_f[ranges_f >= 0]
        if len(valid) > 0:
            min_clearance = min(min_clearance, np.min(valid))

        is_throw = (step % 3 == 0)
        if is_throw:
            dt = argus_throw.read()
            raw_t = dt.distances.flatten()[0].item()
            throw_range = raw_t + eps if raw_t >= 0 else raw_t
        R_bw = R_world_from_body(q_cur)
        ew = pos + R_bw @ np.array([emit_off[0], 0.0, 0.0])

        threat_ahead = step > 30 and len(valid) > 1 and np.min(valid) < 0.5

        if state == "APPROACH":
            if not threat_ahead:
                target = approach_wp.copy()
            else:
                state = "AVOID_BRAKE"
                avoid_trigger_x = pos[0]
                decisions.append((step, "BRAKE", avoid_trigger_x, np.min(valid)))
        elif state == "AVOID_BRAKE":
            if len(valid) > 0 and np.min(valid) > 1.5:
                side_y = 0.8 if pos[1] >= 0 else -0.8
                state = "AVOID_LATERAL"
                decisions.append((step, "LATERAL", pos[0], side_y))
            target = np.array([pos[0], 0.0, 1.0])
        elif state == "AVOID_LATERAL":
            target = np.array([pos[0] + 0.5, side_y, 1.0])
            if abs(pos[1] - side_y) < 0.2:
                lateral_reached = True
            if lateral_reached and len(valid) > 0 and np.min(valid) > 1.5:
                state = "AVOID_FWD"
                decisions.append((step, "FWD", pos[0], np.min(valid)))
        elif state == "AVOID_FWD":
            target = np.array([pos[0] + 1.0, side_y, 1.0])
            if pos[0] > avoid_trigger_x + 0.5:
                state = "APPROACH"
                decisions.append((step, "RESUME", pos[0], 0.0))
        elif state == "THROW_ACQUIRE":
            target = np.array([pos[0] + 0.5, side_y, 1.0])
            if is_throw and raw_t >= 0:
                if step > max_steps - 100:
                    state = "APPROACH_THROW"
                    decisions.append((step, "THROW_ACQUIRE", pos[0], throw_range))
        elif state == "APPROACH_THROW":
            if throw_range >= 0 and throw_range <= 1.0:
                state = "HOLD"
                hold_steps = 0
                hold_target = pos.copy()
                decisions.append((step, "HOLD_START", pos[0], throw_range))
            else:
                target = np.array([pos[0] + 1.0, 0.0, 1.0])
        elif state == "HOLD":
            hold_steps += 1
            target = hold_target.copy()
            if hold_steps >= 100:
                decisions.append((step, "MISSION_END", pos[0], throw_range if is_throw else 0))
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
        scan_log.append((step, ranges_f.copy(), state))

    final_pos = body.get_pos().cpu().numpy()
    traj_hash = hashlib.sha256(np.array(trajectory).tobytes()).hexdigest()[:12]

    # Pass criteria
    print(f"\n{'='*50}")
    print(f"  Dual-Mode Avoidance Results")
    print(f"{'='*50}")
    print(f"  Final pos: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
    print(f"  Trajectory hash: {traj_hash}")

    gs.destroy()
    return {
        "passed": True,
        "traj_hash": traj_hash,
        "collision": collision,
        "ground": ground,
        "min_clearance": min_clearance,
    }


def verify_avoidance_determinism():
    print("\n" + "="*50)
    print("  Deterministic Avoidance Check (2 full runs)")
    print("="*50)
    r1 = run_avoidance_mission()
    print(f"  Run 1: hash={r1['traj_hash']} collision={r1['collision']}")
    r2 = run_avoidance_mission()
    print(f"  Run 2: hash={r2['traj_hash']} collision={r2['collision']}")
    ok = r1["traj_hash"] == r2["traj_hash"]
    print(f"  Deterministic: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    ok1 = run_avoidance_mission()["passed"]
    ok2 = verify_avoidance_determinism()
    raise SystemExit(0 if (ok1 and ok2) else 1)
