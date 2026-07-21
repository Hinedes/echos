"""Dual-mode avoidance mission: FLOOD for obstacle detection, THROW for terminal approach."""
import hashlib

import genesis as gs
import numpy as np

from echos_core import *


FLOOD_AZ_DEG = np.linspace(-30.0, 30.0, 7)
LATERAL_OFFSET = 0.8
MIN_CLEARANCE = 0.20
DESIRED_STANDOFF = 1.0


def choose_avoid_side(ranges, azimuths=FLOOD_AZ_DEG, offset=LATERAL_OFFSET):
    """Choose the side opposite the nearest FLOOD return.

    Positive azimuth points toward +Y in the body frame, so a threat on that
    side is bypassed through -Y.  A centered threat uses aggregate side
    clearance as a deterministic tie-breaker.
    """
    ranges = np.asarray(ranges, dtype=float)
    azimuths = np.asarray(azimuths, dtype=float)
    valid = np.isfinite(ranges) & (ranges >= 0.0)
    if not np.any(valid):
        return float(offset)

    masked = np.where(valid, ranges, np.inf)
    threat_i = int(np.argmin(masked))
    threat_az = azimuths[threat_i]
    if threat_az > 0.0:
        return -float(offset)
    if threat_az < 0.0:
        return float(offset)

    left = masked[azimuths > 0.0]
    right = masked[azimuths < 0.0]
    left_clear = float(np.min(left)) if left.size else np.inf
    right_clear = float(np.min(right)) if right.size else np.inf
    return float(offset) if left_clear >= right_clear else -float(offset)


def run_avoidance_mission(max_steps=1600):
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
        gs.sensors.Raycaster(
            pattern=gs.sensors.SphericalPattern(fov=(60.0, 0.0), n_points=(7, 1)),
            entity_idx=body.idx,
            pos_offset=emit_off,
            euler_offset=(0.0, 0.0, 0.0),
            max_range=3.0,
            no_hit_value=-1.0,
        )
    )
    argus_throw = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
            entity_idx=body.idx,
            pos_offset=emit_off,
            euler_offset=(0.0, 0.0, 0.0),
            max_range=8.0,
            no_hit_value=-1.0,
        )
    )
    scene.build()
    body.set_mass(MASS)
    rigid_solver = scene.sim.rigid_solver
    link_idx = 0

    approach_wp = np.array([2.0, 0.0, 1.0])
    state = "APPROACH"
    target = approach_wp.copy()
    brake_target = None
    lateral_target = None
    forward_target = None
    side_y = 0.0
    lateral_reached = False
    collision = False
    ground = False
    mission_complete = False
    used_throw = False
    hold_target = None
    hold_steps = 0
    final_standoff = None
    last_throw_range = -1.0

    trajectory = []
    scan_log = []
    decisions = []
    min_clearance = float("inf")
    avoid_min_clearance = float("inf")

    try:
        for step in range(max_steps):
            rigid_solver.clear_external_force()
            q_cur = body.get_quat().cpu().numpy()
            pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy()
            omega_w = body.get_ang().cpu().numpy()

            data_f = argus_flood.read()
            raw_f = data_f.distances.cpu().numpy().flatten()
            eps = RAYCAST_ORIGIN - PHYSICAL_EMITTER
            ranges_f = np.where(raw_f >= 0.0, raw_f + eps * np.cos(np.radians(FLOOD_AZ_DEG)), -1.0)
            valid = ranges_f[ranges_f >= 0.0]
            if valid.size:
                min_clearance = min(min_clearance, float(np.min(valid)))

            is_throw = step % 3 == 0
            if is_throw:
                dt = argus_throw.read()
                raw_t = dt.distances.flatten()[0].item()
                last_throw_range = raw_t + eps if raw_t >= 0.0 else -1.0

            threat_ahead = step > 30 and valid.size > 0 and float(np.min(valid)) < 0.5

            if state == "APPROACH":
                target = approach_wp.copy()
                if threat_ahead:
                    side_y = choose_avoid_side(ranges_f)
                    brake_target = pos.copy()
                    brake_target[2] = 1.0
                    state = "AVOID_BRAKE"
                    decisions.append((step, "BRAKE", float(pos[0]), float(np.min(valid))))

            elif state == "AVOID_BRAKE":
                target = brake_target.copy()
                horizontal_speed = float(np.linalg.norm(vel[:2]))
                if horizontal_speed < 0.10:
                    lateral_target = np.array([brake_target[0], side_y, 1.0])
                    state = "AVOID_LATERAL"
                    decisions.append((step, "LATERAL", float(pos[0]), float(side_y)))

            elif state == "AVOID_LATERAL":
                target = lateral_target.copy()
                if abs(pos[1] - side_y) < 0.15 and float(np.linalg.norm(vel[:2])) < 0.35:
                    lateral_reached = True
                    forward_target = np.array([max(pos[0], brake_target[0]) + 1.25, side_y, 1.0])
                    state = "AVOID_FWD"
                    decisions.append((step, "FWD", float(pos[0]), float(side_y)))

            elif state == "AVOID_FWD":
                target = forward_target.copy()
                if valid.size:
                    avoid_min_clearance = min(avoid_min_clearance, float(np.min(valid)))
                if pos[0] >= forward_target[0] - 0.15:
                    state = "THROW_ACQUIRE"
                    decisions.append((step, "THROW_ACQUIRE", float(pos[0]), float(last_throw_range)))

            elif state == "THROW_ACQUIRE":
                target = np.array([pos[0] + 0.5, side_y, 1.0])
                if is_throw and last_throw_range >= 0.0:
                    used_throw = True
                    state = "APPROACH_THROW"
                    decisions.append((step, "THROW_LOCK", float(pos[0]), float(last_throw_range)))

            elif state == "APPROACH_THROW":
                if last_throw_range >= 0.0 and last_throw_range <= DESIRED_STANDOFF:
                    state = "HOLD"
                    hold_steps = 0
                    hold_target = pos.copy()
                    hold_target[2] = 1.0
                    final_standoff = float(last_throw_range)
                    decisions.append((step, "HOLD_START", float(pos[0]), float(last_throw_range)))
                    target = hold_target.copy()
                else:
                    advance = 0.5
                    if last_throw_range >= 0.0:
                        advance = min(0.5, max(0.05, last_throw_range - DESIRED_STANDOFF))
                    target = np.array([pos[0] + advance, side_y, 1.0])

            elif state == "HOLD":
                hold_steps += 1
                target = hold_target.copy()
                if is_throw and last_throw_range >= 0.0:
                    final_standoff = float(last_throw_range)
                if hold_steps >= 100:
                    mission_complete = True
                    decisions.append((step, "MISSION_END", float(pos[0]), float(final_standoff or -1.0)))
                    break

            if valid.size and state in ("AVOID_BRAKE", "AVOID_LATERAL", "AVOID_FWD"):
                avoid_min_clearance = min(avoid_min_clearance, float(np.min(valid)))

            T_total, q_des = position_pd(target, pos, vel)
            tau, _ = attitude_pd(q_des, q_cur, omega_w)
            thrusts, _, _, _ = mixer_with_authority(T_total, tau[0], tau[1], tau[2])
            apply_rotor_forces(rigid_solver, link_idx, thrusts)
            scene.step()

            pos = body.get_pos().cpu().numpy()
            if pos[2] <= BODY_H / 2 + 0.005:
                ground = True
            if check_collision(body):
                collision = True
            trajectory.append(pos.copy())
            scan_log.append((step, ranges_f.copy(), state))

            if collision or ground:
                break

        final_pos = body.get_pos().cpu().numpy()
        traj_hash = hashlib.sha256(np.asarray(trajectory).tobytes()).hexdigest()[:12]
        required_events = {"BRAKE", "LATERAL", "FWD", "THROW_ACQUIRE", "THROW_LOCK", "HOLD_START", "MISSION_END"}
        observed_events = {d[1] for d in decisions}
        event_sequence_ok = required_events.issubset(observed_events)
        clearance_ok = np.isfinite(min_clearance) and min_clearance > MIN_CLEARANCE
        avoid_clearance_ok = np.isfinite(avoid_min_clearance) and avoid_min_clearance > MIN_CLEARANCE
        standoff_ok = final_standoff is not None and 0.75 <= final_standoff <= DESIRED_STANDOFF + 0.10
        passed = all(
            [
                mission_complete,
                lateral_reached,
                used_throw,
                event_sequence_ok,
                not collision,
                not ground,
                clearance_ok,
                avoid_clearance_ok,
                standoff_ok,
            ]
        )

        print(f"\n{'='*50}")
        print("  Dual-Mode Avoidance Results")
        print(f"{'='*50}")
        print(f"  Final state: {state}")
        print(f"  Final pos: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
        print(f"  Final standoff: {final_standoff}")
        print(f"  Minimum clearance: {min_clearance:.4f}")
        print(f"  Avoidance clearance: {avoid_min_clearance:.4f}")
        print(f"  Collision: {collision}, ground: {ground}")
        print(f"  Events: {[d[1] for d in decisions]}")
        print(f"  Trajectory hash: {traj_hash}")
        print(f"  DUAL-MODE AVOIDANCE {'PASS' if passed else 'FAIL'}")

        return {
            "passed": passed,
            "traj_hash": traj_hash,
            "collision": collision,
            "ground": ground,
            "min_clearance": min_clearance,
            "avoid_min_clearance": avoid_min_clearance,
            "final_standoff": final_standoff,
            "final_state": state,
            "events": [d[1] for d in decisions],
        }
    finally:
        gs.destroy()


def verify_avoidance_determinism():
    print("\n" + "="*50)
    print("  Deterministic Avoidance Check (2 full runs)")
    print("="*50)
    r1 = run_avoidance_mission()
    print(f"  Run 1: pass={r1['passed']} hash={r1['traj_hash']} collision={r1['collision']}")
    r2 = run_avoidance_mission()
    print(f"  Run 2: pass={r2['passed']} hash={r2['traj_hash']} collision={r2['collision']}")
    ok = (
        r1["passed"]
        and r2["passed"]
        and r1["traj_hash"] == r2["traj_hash"]
        and r1["events"] == r2["events"]
        and r1["final_state"] == r2["final_state"]
    )
    print(f"  Deterministic: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify_avoidance_determinism() else 1)
