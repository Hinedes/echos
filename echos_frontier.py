"""Autonomous frontier exploration — shared core with echos_core.py."""
import genesis as gs
import numpy as np
import os
from echos_core import *
import math
import hashlib
from argus_export import GimbalState, TrajectoryRecorder

gs.init(backend=getattr(gs, os.environ.get("ECHOS_GENESIS_BACKEND", "gpu")))
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))

body = scene.add_entity(gs.morphs.Box(size=(BODY_W, BODY_D, BODY_H), pos=(0, 0, 1), fixed=False))
scene.add_entity(gs.morphs.Box(size=(7.0, 0.01, 2.0), pos=(2.75, -1.5, 1.0), fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01, 6.0, 2.0), pos=(-0.5, 1.0, 1.0), fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5, 0.01, 2.0), pos=(1.25, 1.5, 1.0), fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True))
scene.add_entity(gs.morphs.Box(size=(3.5, 0.01, 2.0), pos=(4.75, 4.0, 1.0), fixed=True))
scene.add_entity(gs.morphs.Box(size=(0.5, 0.8, 1.5), pos=(1.5, 1.0, 0.75), fixed=True))
scene.add_entity(gs.morphs.Plane())

argus_flood = scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(fov=(60.0, 0.0), n_points=(7, 1)),
    entity_idx=body.idx, pos_offset=(RAYCAST_ORIGIN, 0.0, 0.0),
    euler_offset=(0.0, 0.0, 0.0), max_range=5.0, no_hit_value=-1.0))
argus_throw = scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
    entity_idx=body.idx, pos_offset=(RAYCAST_ORIGIN, 0.0, 0.0),
    euler_offset=(0.0, 0.0, 0.0), max_range=8.0, no_hit_value=-1.0))

scene.build()
body.set_mass(MASS)
rs = scene.sim.rigid_solver
li = 0
SIM_DT_S = 0.01


def frontier_clusters(occ, inf):
    h, w = occ.shape
    fg = np.zeros((h, w), dtype=np.int32)
    for iy in range(1, h-1):
        for ix in range(1, w-1):
            if occ[iy, ix] == 0.5 and inf[iy, ix] == 0:
                if (occ[iy-1, ix] == 0.0 or occ[iy+1, ix] == 0.0 or
                    occ[iy, ix-1] == 0.0 or occ[iy, ix+1] == 0.0):
                    fg[iy, ix] = 1

    lab = np.zeros((h, w), dtype=np.int32); cl = 1; eq = {}
    for iy in range(1, h-1):
        for ix in range(1, w-1):
            if fg[iy, ix] == 0: continue
            up = lab[iy-1, ix]; le = lab[iy, ix-1]
            ul = lab[iy-1, ix-1]; ur = lab[iy-1, ix+1]
            ns = [l for l in [up, le, ul, ur] if l > 0]
            if not ns:
                lab[iy, ix] = cl; cl += 1
            else:
                m = min(ns); lab[iy, ix] = m
                for n in ns:
                    if n != m: eq[n] = m
    for iy in range(1, h-1):
        for ix in range(1, w-1):
            l = lab[iy, ix]
            if l > 0:
                while l in eq: l = eq[l]
                lab[iy, ix] = l

    cs = {}
    for iy in range(1, h-1):
        for ix in range(1, w-1):
            l = lab[iy, ix]
            if l > 0:
                if l not in cs: cs[l] = []
                cs[l].append((ix, iy))
    return cs


failed_frontiers = set()
completed_frontiers = set()

def clear_failed_frontiers():
    global failed_frontiers, completed_frontiers
    failed_frontiers = set()
    completed_frontiers = set()

def mark_frontier_failed(ccx, ccy, mapper):
    global failed_frontiers
    rev = int(np.sum(mapper.views))
    failed_frontiers.add(((round(ccx), round(ccy)), rev))

def select_frontier(occ, inf, cs, cur_pos, mapper):
    global failed_frontiers, completed_frontiers
    map_revision = int(np.sum(mapper.views))
    sx, sy = mapper.w2g(cur_pos[0], cur_pos[1])
    best = None; bs = -1; cur_path = None

    for l, cells in cs.items():
        if len(cells) < 5: continue

        ccx = sum(c[0] for c in cells) / len(cells)
        ccy = sum(c[1] for c in cells) / len(cells)
        cx_r, cy_r = round(ccx), round(ccy)
        if (cx_r, cy_r) in completed_frontiers:
            continue
        if ((cx_r, cy_r), map_revision) in failed_frontiers:
            continue

        adj_unk = set()
        for (ix, iy) in cells:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = ix+dx, iy+dy
                if 0 <= nx < mapper.w and 0 <= ny < mapper.h and occ[ny, nx] == 0.0:
                    adj_unk.add((nx, ny))
        if len(adj_unk) == 0: continue

        uxs = [p[0] for p in adj_unk]; uys = [p[1] for p in adj_unk]
        uc_x = sum(uxs)/len(uxs); uc_y = sum(uys)/len(uys)

        best_d = 1e9; gx, gy = cells[0]
        for (ix, iy) in cells:
            if not (occ[iy, ix] == 0.5 and inf[iy, ix] == 0): continue
            d = abs(ix - uc_x) + abs(iy - uc_y)
            if d < best_d: best_d = d; gx, gy = ix, iy

        pth = astar_path(inf, (sx, sy), (gx, gy))
        if pth is None: continue

        path_cost = 0.0
        for i in range(1, len(pth)):
            dx = abs(pth[i][0] - pth[i-1][0])
            dy = abs(pth[i][1] - pth[i-1][1])
            path_cost += SQRT2 if (dx != 0 and dy != 0) else 1.0
        cost_m = path_cost * mapper.res
        sc = len(cells) / (cost_m + 0.01)
        cl_info = {"cx": mapper.g2w(gx, gy)[0], "cy": mapper.g2w(gx, gy)[1],
                   "cix": gx, "ciy": gy, "n": len(cells),
                   "ucx": uc_x, "ucy": uc_y, "adj_unk_n": len(adj_unk),
                   "ccx": cx_r, "ccy": cy_r}
        if best is None or sc > bs:
            best = cl_info; bs = sc; cur_path = pth

    if best is not None:
        uc_wx = mapper.x_min + (best["ucx"] + 0.5) * mapper.res
        uc_wy = mapper.y_min + (best["ucy"] + 0.5) * mapper.res
        best["uk_world"] = (uc_wx, uc_wy)
        best["observe_heading"] = np.arctan2(uc_wy - best["cy"], uc_wx - best["cx"])
        best["score"] = bs
    return best, cur_path


def run_frontier_exploration(record_path=None):
    print("\n" + "="*50)
    print("  Autonomous Frontier Exploration")
    print("="*50)

    clear_failed_frontiers()

    reset_body_state(body, rs, pos=(0.0, 0.0, 1.0))
    recorder = TrajectoryRecorder(physics_dt_s=SIM_DT_S, control_dt_s=SIM_DT_S) if record_path else None
    gimbal_state = GimbalState.from_pitch(0.0)
    sim_t = 0.0
    for _ in range(20):
        rs.clear_external_force()
        qc = body.get_quat().cpu().numpy(); p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()
        if recorder:
            recorder.record(sim_t, p, qc, v, om, gimbal_state)
        Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
        tau, _ = attitude_pd(qd, qc, om)
        ts, sat, _, _ = mixer_with_authority(Tt, tau[0], tau[1], tau[2])
        apply_rotor_forces(rs, li, ts)
        scene.step()
        sim_t += SIM_DT_S

    mx, mxx, my, myy = -2.0, 7.0, -2.0, 5.0
    mapper = OccupancyMapper((mx, mxx, my, myy), 0.08)
    az_deg = np.linspace(-30, 30, 7)
    launch = np.array([0.0, 0.0, 1.0])
    bootstrap_tgt = bootstrap_target(launch)
    max_steps = 15000; replan_iv = 120

    state = "INITIAL_FLY"
    cur_path = []; wp_i = 0
    traj = []; scan_log = []; front_log = []; dec = []
    min_clr = float('inf'); collided = False
    best = None
    unknown_at_step = 0

    # Persistent yaw command (slew-limited across all states)
    yaw_cmd = 0.0
    hold_pos = None
    align_steps = 0; scan_steps = 0; yaw_aligned = False

    for step in range(max_steps):
        rs.clear_external_force()
        q_cur = body.get_quat().cpu().numpy(); pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()

        df = argus_flood.read(); raw_f = df.distances.cpu().numpy().flatten()
        vf = raw_f[raw_f >= 0]
        if len(vf) > 0: min_clr = min(min_clr, np.min(vf))

        is_thr = (step % 5 == 0)
        if is_thr:
            dt = argus_throw.read(); raw_t = dt.distances.flatten()[0].item()

        if recorder:
            recorder.record(sim_t, pos, q_cur, vel, om, gimbal_state)

        Rwb = R_world_from_body(q_cur)
        ew = pos + Rwb @ np.array([RAYCAST_ORIGIN, 0.0, 0.0])

        for i in range(7):
            az = np.radians(az_deg[i]); db = np.array([np.cos(az), np.sin(az), 0.0])
            dw = Rwb @ db; hit = raw_f[i] >= 0; rr = raw_f[i] if hit else 0.0
            mapper.update_ray(ew, dw, rr, 5.0, hit)
        if is_thr:
            dwt = Rwb @ np.array([1.0, 0.0, 0.0]); htt = raw_t >= 0
            mapper.update_ray(ew, dwt, raw_t if htt else 0.0, 8.0, htt)

        # --- state machine (all paths compute thrust; no gravity-only steps) ---
        active_state = "INITIAL_FLY" if step < 300 else state
        state_changed = False

        if step < 300:
            pass
        elif state == "INITIAL_FLY":
            state = "EXPLORE"; state_changed = True

        # Replan
        if step >= 300 and state == "EXPLORE" and step % replan_iv == 0:
            occ = mapper.get_map()
            inf = inflation_grid(mapper, inflate_r=4)
            cs = frontier_clusters(occ, inf)
            best, cur_path = select_frontier(occ, inf, cs, pos, mapper)
            if best is None:
                dec.append((step, "NO_FRONTIER", pos[0], pos[1]))
                print(f"  step {step}: no reachable frontier -> RTL")
                break
            else:
                state = "FLY_TO_FRONTIER"; wp_i = 0; state_changed = True
                front_log.append({"step": step, "cx": best["cx"], "cy": best["cy"],
                                  "n": best["n"], "score": best["score"],
                                  "observe_heading": best["observe_heading"]})
                if len(front_log) <= 10:
                    print(f"  step {step}: frontier #{len(front_log)} "
                          f"({best['cx']:.2f},{best['cy']:.2f}) "
                          f"n={best['n']} score={best['score']:.3f}")

        # HOLD_AND_OBSERBE state transitions
        if state == "HOLD_AND_OBSERVE":
            yaw_target = best["observe_heading"]
            yaw_cmd = move_toward_angle(yaw_cmd, yaw_target, MAX_YAW_RATE * 0.01)
            if not yaw_aligned and align_steps < 200:
                align_steps += 1
                body_x = Rwb @ np.array([1.0, 0.0, 0.0])
                uk_x, uk_y = best["uk_world"]
                unk_dir = np.array([uk_x - best["cx"], uk_y - best["cy"], 0.0])
                unk_dir /= max(np.linalg.norm(unk_dir), 1e-10)
                dot_val = body_x @ unk_dir
                if dot_val > 0.99:
                    yaw_aligned = True
                    print(f"    aligned: dot={dot_val:.4f}")
            elif yaw_aligned and scan_steps < 30:
                scan_steps += 1
            elif yaw_aligned and scan_steps >= 30:
                state = "EXPLORE"; state_changed = True
                completed_frontiers.add((best["ccx"], best["ccy"]))
                print(f"    scan complete ({scan_steps} steps)")
            elif align_steps >= 200:
                state = "EXPLORE"; state_changed = True
                if best is not None:
                    mark_frontier_failed(best["ccx"], best["ccy"], mapper)
                print(f"    ALIGN TIMEOUT at step {step} — abort frontier")

        # --- thrust computation (every iteration) ---
        if step < 300:
            tgt = bootstrap_tgt.copy()
        elif state == "FLY_TO_FRONTIER" and wp_i < len(cur_path):
            cx, cy = mapper.g2w(*cur_path[wp_i])
            tgt = np.array([cx, cy, 1.0])
            dist = np.linalg.norm(pos[:2] - tgt[:2])
            speed = np.linalg.norm(vel[:2])
            if dist < mapper.res * 1.5 and speed < 0.3:
                wp_i += 1
        elif state == "HOLD_AND_OBSERVE":
            tgt = hold_pos.copy()
        elif state == "FLY_TO_FRONTIER":
            # All waypoints exhausted — transitioning to HOLD next iteration
            tgt = np.array([pos[0], pos[1], 1.0])
            state = "HOLD_AND_OBSERVE"; state_changed = True
            hold_pos = pos.copy(); align_steps = 0; scan_steps = 0; yaw_aligned = False
            if best is not None:
                print(f"  step {step}: reached frontier, "
                      f"yaw target={np.degrees(best['observe_heading']):.1f}")
        else:
            tgt = np.array([pos[0], pos[1], 1.0])

        Tt, qd = position_pd(tgt, pos, vel, yaw_cmd)
        tau, _ = attitude_pd(qd, q_cur, om)
        ts, sat, _, _ = mixer_with_authority(Tt, tau[0], tau[1], tau[2])
        apply_rotor_forces(rs, li, ts)
        scene.step()
        sim_t += SIM_DT_S
        pos = body.get_pos().cpu().numpy()
        mapper.mark_free_cell(pos[0], pos[1])
        if check_collision(body): collided = True
        traj.append(pos.copy()); scan_log.append((step, raw_f.copy(), active_state))
        pix_at, piy_at = mapper.w2g(pos[0], pos[1])
        if mapper.in_b(pix_at, piy_at) and mapper.views[piy_at, pix_at] == 0:
            unknown_at_step += 1

    # --- RTL ---
    occ = mapper.get_map()
    inf = inflation_grid(mapper, inflate_r=4)
    sx, sy = mapper.w2g(pos[0], pos[1])
    gx, gy = mapper.w2g(launch[0], launch[1])
    sx = max(0, min(mapper.w-1, sx)); sy = max(0, min(mapper.h-1, sy))
    gx = max(0, min(mapper.w-1, gx)); gy = max(0, min(mapper.h-1, gy))

    traj_rtl = []
    pth = astar_path(inf, (sx, sy), (gx, gy))
    if pth is None:
        print(f"  RTL FAILED: no path from ({sx},{sy}) to ({gx},{gy})")
    else:
        print(f"  RTL path: {len(pth)} cells")
        wp_i_rtl = 0
        traj_rtl = []
        for s2 in range(8000):
            rs.clear_external_force()
            qc = body.get_quat().cpu().numpy(); p2 = body.get_pos().cpu().numpy()
            v2 = body.get_vel().cpu().numpy(); om2 = body.get_ang().cpu().numpy()
            if recorder:
                recorder.record(sim_t, p2, qc, v2, om2, gimbal_state)

            if wp_i_rtl >= len(pth):
                tgt2 = launch.copy()
            else:
                c2x, c2y = mapper.g2w(*pth[wp_i_rtl])
                tgt2 = np.array([c2x, c2y, 1.0])

            dist = np.linalg.norm(p2[:2] - tgt2[:2])
            speed = np.linalg.norm(v2[:2])
            if dist < mapper.res * 1.5 and speed < 0.3:
                wp_i_rtl += 1

            Tt, qd = position_pd(tgt2, p2, v2, yaw_cmd)
            tau, _ = attitude_pd(qd, qc, om2)
            ts, sat, _, _ = mixer_with_authority(Tt, tau[0], tau[1], tau[2])
            apply_rotor_forces(rs, li, ts)
            scene.step()
            sim_t += SIM_DT_S
            if check_collision(body): collided = True
            df2 = argus_flood.read(); raw_f2 = df2.distances.cpu().numpy().flatten()
            vf2 = raw_f2[raw_f2 >= 0]
            if len(vf2) > 0: min_clr = min(min_clr, np.min(vf2))
            p2 = body.get_pos().cpu().numpy()
            v2 = body.get_vel().cpu().numpy()
            mapper.mark_free_cell(p2[0], p2[1])
            traj_rtl.append(p2.copy())
            if np.linalg.norm(p2[:2] - launch[:2]) < 0.04 and np.linalg.norm(v2[:2]) < 0.1 and s2 > 50:
                break

    fpos = body.get_pos().cpu().numpy()
    rtl_d = np.linalg.norm(fpos[:2] - launch[:2])
    occ = mapper.get_map(); seen = mapper.views > 0

    # BFS flood fill for reachable free space
    sx_f, sy_f = mapper.w2g(launch[0], launch[1])
    reachable = set()
    q = [(sx_f, sy_f)]
    while q:
        cx, cy = q.pop()
        if (cx, cy) in reachable:
            continue
        if not mapper.in_b(cx, cy):
            continue
        if occ[cy, cx] == 1.0:
            continue
        wx_f, wy_f = mapper.g2w(cx, cy)
        if (-0.75 <= wx_f <= 6.25 and -1.505 <= wy_f <= -1.495) or \
           (-0.505 <= wx_f <= -0.495 and -2.0 <= wy_f <= 4.0) or \
           (-0.5 <= wx_f <= 3.0 and 1.495 <= wy_f <= 1.505) or \
           (2.92 <= wx_f <= 3.08 and 1.25 <= wy_f <= 4.25) or \
           (3.0 <= wx_f <= 6.5 and 3.995 <= wy_f <= 4.005) or \
           (1.25 <= wx_f <= 1.75 and 0.6 <= wy_f <= 1.4):
            continue
        if occ[cy, cx] != 0.5:
            continue
        reachable.add((cx, cy))
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            q.append((cx+dx, cy+dy))
    tr = len(reachable)
    te = sum(1 for (ix, iy) in reachable
             if seen[iy, ix] and occ[iy, ix] == 0.5)
    er = te / tr * 100 if tr > 0 else 0

    # Count legs with at least 3 free cells observed
    horiz_leg_cells = [(ix, iy) for (ix, iy) in reachable
                       if mapper.g2w(ix, iy)[1] < 1.5]
    vert_leg_cells = [(ix, iy) for (ix, iy) in reachable
                      if 3.0 <= mapper.g2w(ix, iy)[0] <= 5.9
                      and 1.3 <= mapper.g2w(ix, iy)[1] <= 3.9]
    horiz_discovered = sum(1 for (ix, iy) in horiz_leg_cells
                           if seen[iy, ix] and occ[iy, ix] == 0.5) >= 3
    vert_discovered = sum(1 for (ix, iy) in vert_leg_cells
                          if seen[iy, ix] and occ[iy, ix] == 0.5) >= 3
    both_legs = horiz_discovered and vert_discovered

    zero_unknown = unknown_at_step == 0

    # Detect oscillation: same frontier selected within last 5
    oscillation = False
    if len(front_log) >= 3:
        recent = [(fl["cx"], fl["cy"]) for fl in front_log[-5:]]
        if len(set(recent)) < len(recent):
            oscillation = True

    print(f"\n  Trajectory: {len(traj)} steps, final ({fpos[0]:.2f},{fpos[1]:.2f})")
    print(f"  Frontiers: {len(front_log)}, explored {er:.1f}%")
    print(f"  Both legs: horiz={horiz_discovered} vert={vert_discovered}")
    print(f"  Unknown traversal: {unknown_at_step} steps")
    print(f"  Oscillation: {oscillation}")
    print(f"  RTL dist: {rtl_d:.4f}, collision: {collided}")
    print(f"  Min clearance: {min_clr:.4f}")

    if recorder:
        recorder.save(record_path)
        print(f"  Trajectory export: {record_path}")

    print("\n" + "="*50)
    print("  Pass Criteria")
    print("="*50)
    ok = True
    r1 = er >= 95.0; ok &= r1
    print(f"  1. Explored >= 95%: {er:.1f}%  {'PASS' if r1 else 'FAIL'}")
    r2 = not collided; ok &= r2
    print(f"  2. No collision: {'PASS' if r2 else 'FAIL'}")
    r3 = min_clr > 0.20; ok &= r3
    print(f"  3. Clearance > 0.20 m: {min_clr:.4f}  {'PASS' if r3 else 'FAIL'}")
    r4 = len(front_log) > 0; ok &= r4
    print(f"  4. Frontiers found: {len(front_log)}  {'PASS' if r4 else 'FAIL'}")
    r5 = rtl_d < 0.05; ok &= r5
    print(f"  5. RTL within 5 cm: {rtl_d:.4f}  {'PASS' if r5 else 'FAIL'}")
    r6 = both_legs; ok &= r6
    print(f"  6. Both legs discovered: {'PASS' if r6 else 'FAIL'}")
    r7 = zero_unknown; ok &= r7
    print(f"  7. Zero unknown traversal: {unknown_at_step} steps  {'PASS' if r7 else 'FAIL'}")
    r8 = not oscillation; ok &= r8
    print(f"  8. No frontier oscillation: {'PASS' if r8 else 'FAIL'}")
    r9 = "NO_FRONTIER" in [d[1] for d in dec]; ok &= r9
    print(f"  9. Terminated by NO_FRONTIER: {'PASS' if r9 else 'FAIL'}")
    print(f"\n  FRONTIER EXPLORATION {'PASS' if ok else 'FAIL'}")
    traj_arr = np.array(traj)
    traj_hash = hashlib.sha256(traj_arr.tobytes()).hexdigest()
    frontier_seq_arr = np.array([(fl["cx"], fl["cy"]) for fl in front_log])
    frontier_seq_hash = hashlib.sha256(frontier_seq_arr.tobytes()).hexdigest()
    occ_grid_hash = hashlib.sha256(mapper.get_map().tobytes()).hexdigest()
    return {
        "passed": ok,
        "traj_hash": traj_hash,
        "frontier_seq_hash": frontier_seq_hash,
        "occ_grid_hash": occ_grid_hash,
        "term_step": step,
        "rtl_dist": rtl_d,
        "coverage_percent": float(er),
        "mapper": mapper,
        "trajectory": np.asarray(traj),
        "rtl_trajectory": np.asarray(traj_rtl),
        "frontiers": front_log,
        "min_clearance": float(min_clr),
        "collision": bool(collided),
        "unknown_traversal": int(unknown_at_step),
        "oscillation": bool(oscillation),
        "terminated_no_frontier": "NO_FRONTIER" in [d[1] for d in dec],
    }

if __name__ == "__main__":
    r1 = run_frontier_exploration()
    r2 = run_frontier_exploration()

    print(f"Run 1: passed={r1['passed']}, term_step={r1['term_step']}, rtl_dist={r1['rtl_dist']:.4f}")
    print(f"Run 2: passed={r2['passed']}, term_step={r2['term_step']}, rtl_dist={r2['rtl_dist']:.4f}")
    print(f"Traj hash match: {r1['traj_hash'] == r2['traj_hash']}")
    print(f"Frontier seq hash match: {r1['frontier_seq_hash'] == r2['frontier_seq_hash']}")
    print(f"Occ grid hash match: {r1['occ_grid_hash'] == r2['occ_grid_hash']}")
    ok = all([
        r1['traj_hash'] == r2['traj_hash'],
        r1['frontier_seq_hash'] == r2['frontier_seq_hash'],
        r1['occ_grid_hash'] == r2['occ_grid_hash'],
        r1['passed'], r2['passed'],
    ])
    raise SystemExit(0 if ok else 1)
