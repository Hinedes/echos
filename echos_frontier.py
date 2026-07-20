"""Autonomous frontier exploration — shared core with echos_core.py."""
import genesis as gs
import numpy as np
from echos_core import *

gs.init(backend=gs.amdgpu)
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
    entity_idx=body.idx, pos_offset=EMIT_OFFSET, euler_offset=(0.0, 0.0, 0.0),
    max_range=5.0, no_hit_value=-1.0))
argus_throw = scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
    entity_idx=body.idx, pos_offset=EMIT_OFFSET, euler_offset=(0.0, 0.0, 0.0),
    max_range=8.0, no_hit_value=-1.0))

scene.build()
body.set_mass(MASS)
rs = scene.sim.rigid_solver
li = 0

def reset_body(pos=(0.0, 0.0, 1.0), quat=None):
    body.set_pos(pos)
    body.set_quat(quat if quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
    rs.clear_external_force()


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
            if fg[iy, ix] == 0:
                continue
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


def select_frontier(occ, inf, cs, pos, mapper, gx, gy):
    sx, sy = mapper.w2g(pos[0], pos[1])
    best = None; bs = -1; bk = None; cur_path = None

    for l, cells in cs.items():
        if len(cells) < 5: continue

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

        cx_w, cy_w = mapper.g2w(gx, gy)
        gn = len(cells); ct = len(pth) * mapper.res
        sc = gn / (ct + 0.01)
        cl_info = {"cx": cx_w, "cy": cy_w, "cix": gx, "ciy": gy, "n": gn,
                   "ucx": uc_x, "ucy": uc_y, "adj_unk_n": len(adj_unk)}
        ky = (sc, gx, gy)
        if best is None or sc > bs or (abs(sc-bs) < 1e-6 and ky < bk):
            best = cl_info; bs = sc; bk = ky; cur_path = pth

    if best is not None:
        uc_wx = mapper.x_min + (best["ucx"] + 0.5) * mapper.res
        uc_wy = mapper.y_min + (best["ucy"] + 0.5) * mapper.res
        best["uk_world"] = (uc_wx, uc_wy)
        best["observe_heading"] = np.arctan2(uc_wy - best["cy"], uc_wx - best["cx"])
        best["score"] = bs
    return best, cur_path


def run_frontier_exploration():
    print("\n" + "="*50)
    print("  Autonomous Frontier Exploration")
    print("="*50)

    reset_body(pos=(0.0, 0.0, 1.0))
    for _ in range(20):
        rs.clear_external_force()
        qc = body.get_quat().cpu().numpy(); p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()
        Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
        apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, qc, om)[0])[0], 0.0, None))
        scene.step()

    mx, mxx, my, myy = -2.0, 7.0, -2.0, 5.0
    mapper = OccupancyMapper((mx, mxx, my, myy), 0.08)
    eps = 0.004; az_deg = np.linspace(-30, 30, 7)
    launch = np.array([0.0, 0.0, 1.0])
    max_steps = 15000; replan_iv = 120

    state = "INITIAL_FLY"
    cur_path = []; wp_i = 0
    traj = []; scan_log = []; front_log = []; dec = []
    min_clr = float('inf'); collided = False
    best = None

    # HOLD_AND_OBSERVE persistent state
    hold_pos = None
    hold_yaw = 0.0
    align_steps = 0
    scan_steps = 0
    yaw_aligned = False
    scanning = False

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

        Rwb = R_world_from_body(q_cur)
        ew = pos + Rwb @ np.array([EMIT_OFFSET[0], 0.0, 0.0])

        for i in range(7):
            az = np.radians(az_deg[i]); db = np.array([np.cos(az), np.sin(az), 0.0])
            dw = Rwb @ db; hit = raw_f[i] >= 0; rr = raw_f[i] if hit else 0.0
            mapper.update_ray(ew, dw, rr, 5.0, hit, eps)
        if is_thr:
            dwt = Rwb @ np.array([1.0, 0.0, 0.0]); htt = raw_t >= 0
            mapper.update_ray(ew, dwt, raw_t if htt else 0.0, 8.0, htt, eps)

        # State machine
        if step < 300:
            tgt = np.array([pos[0]+2.0, 0.0, 1.0])
            Tt, qd = position_pd(tgt, pos, vel)
            apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)[0])[0], 0.0, None))
            scene.step()
            pos = body.get_pos().cpu().numpy()
            if check_collision(body): collided = True
            traj.append(pos.copy()); scan_log.append((step, raw_f.copy(), "INITIAL_FLY"))
            continue

        if state == "INITIAL_FLY":
            state = "EXPLORE"

        # ---- EXPLORE / replan ----
        if state == "EXPLORE" and step % replan_iv == 0:
            occ = mapper.get_map()
            inf = inflation_grid(mapper, inflate_r=4)

            sx, sy = mapper.w2g(pos[0], pos[1])
            cs = frontier_clusters(occ, inf)
            best, cur_path = select_frontier(occ, inf, cs, pos, mapper, sx, sy)

            if best is None:
                dec.append((step, "NO_FRONTIER", pos[0], pos[1]))
                print(f"  step {step}: no reachable frontier -> RTL")
                # Break immediately — start RTL from current airborne state
                break
            else:
                state = "FLY_TO_FRONTIER"
                wp_i = 0
                front_log.append({"step": step, "cx": best["cx"], "cy": best["cy"],
                                  "n": best["n"], "score": best["score"],
                                  "observe_heading": best["observe_heading"]})
                if len(front_log) <= 10:
                    print(f"  step {step}: frontier #{len(front_log)} ({best['cx']:.2f},{best['cy']:.2f}) "
                          f"n={best['n']} score={best['score']:.3f}")

        # ---- FLY_TO_FRONTIER ----
        if state == "FLY_TO_FRONTIER":
            if wp_i >= len(cur_path):
                state = "HOLD_AND_OBSERVE"
                hold_pos = pos.copy()
                hold_yaw = best["observe_heading"]
                align_steps = 0; scan_steps = 0
                yaw_aligned = False; scanning = False
                if best is not None:
                    print(f"  step {step}: reached frontier, yaw target={np.degrees(hold_yaw):.1f}")
            else:
                cx, cy = mapper.g2w(*cur_path[wp_i])
                tgt = np.array([cx, cy, 1.0])
                if np.linalg.norm(pos[:2] - tgt[:2]) < 0.25:
                    wp_i += 1
                Tt, qd = position_pd(tgt, pos, vel)
                apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)[0])[0], 0.0, None))

        # ---- HOLD_AND_OBSERVE ----
        elif state == "HOLD_AND_OBSERVE":
            if not yaw_aligned and align_steps < 200:
                psi_des = -hold_yaw
                # Use captured hold position
                Tt, qd = position_pd(hold_pos, pos, vel, psi_des)
                apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)[0])[0], 0.0, None))
                align_steps += 1

                body_x = Rwb @ np.array([1.0, 0.0, 0.0])
                uk_x, uk_y = best["uk_world"]
                unk_dir = np.array([uk_x - best["cx"], uk_y - best["cy"], 0.0])
                unk_dir /= np.linalg.norm(unk_dir)
                dot_val = body_x @ unk_dir

                if align_steps == 1:
                    print(f"    yaw aligning... heading={np.degrees(psi_des):.1f} dot={dot_val:.4f}")
                if dot_val > 0.99:
                    yaw_aligned = True
                    print(f"    aligned: dot={dot_val:.4f}  scan begins")
            elif yaw_aligned and scan_steps < 30:
                # Scan using captured hold position AND captured hold yaw
                Tt, qd = position_pd(hold_pos, pos, vel, -hold_yaw)
                apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)[0])[0], 0.0, None))
                scan_steps += 1
            elif yaw_aligned and scan_steps >= 30:
                state = "EXPLORE"
                print(f"    scan complete ({scan_steps} steps)")
            else:
                # Fallthrough: hold position with observation yaw
                Tt, qd = position_pd(hold_pos, pos, vel, -hold_yaw)
                apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)[0])[0], 0.0, None))

        # ---- EXPLORE (idle between replans) ----
        elif state == "EXPLORE":
            Tt, qd = position_pd(np.array([pos[0], pos[1], 1.0]), pos, vel)
            apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)[0])[0], 0.0, None))

        scene.step()
        pos = body.get_pos().cpu().numpy()
        if check_collision(body): collided = True
        traj.append(pos.copy()); scan_log.append((step, raw_f.copy(), state))

    # ---- RTL ----
    occ = mapper.get_map()
    inf = inflation_grid(mapper, inflate_r=4)
    sx, sy = mapper.w2g(pos[0], pos[1])
    gx, gy = mapper.w2g(launch[0], launch[1])
    sx = max(0, min(mapper.w-1, sx)); sy = max(0, min(mapper.h-1, sy))
    gx = max(0, min(mapper.w-1, gx)); gy = max(0, min(mapper.h-1, gy))

    pth = astar_path(inf, (sx, sy), (gx, gy))
    if pth is None:
        print(f"  RTL FAILED: no path from ({sx},{sy}) to ({gx},{gy})")
    else:
        print(f"  RTL path: {len(pth)} cells")
        wp_i_rtl = 0
        for s2 in range(8000):
            rs.clear_external_force()
            qc = body.get_quat().cpu().numpy(); p2 = body.get_pos().cpu().numpy()
            v2 = body.get_vel().cpu().numpy(); om2 = body.get_ang().cpu().numpy()

            if wp_i_rtl >= len(pth):
                tgt2 = launch.copy()
            else:
                c2x, c2y = mapper.g2w(*pth[wp_i_rtl])
                tgt2 = np.array([c2x, c2y, 1.0])

            if np.linalg.norm(p2[:2] - tgt2[:2]) < 0.25:
                wp_i_rtl += 1

            if np.linalg.norm(p2[:2] - launch[:2]) < 0.04 and s2 > 50:
                break

            Tt, qd = position_pd(tgt2, p2, v2)
            apply_rotor_forces(rs, li, np.clip(mixer(Tt, *attitude_pd(qd, qc, om2)[0])[0], 0.0, None))
            scene.step()

    fpos = body.get_pos().cpu().numpy()
    rtl_d = np.linalg.norm(fpos[:2] - launch[:2])
    occ = mapper.get_map(); seen = mapper.views > 0

    tp = 0; fp_val = 0
    for iy in range(mapper.h):
        for ix in range(mapper.w):
            wx, wy = mapper.g2w(ix, iy)
            if not seen[iy, ix]: continue
            in_corridor = (-0.3 <= wx <= 5.9 and -1.3 <= wy <= 1.3) or \
                          (3.1 <= wx <= 5.9 and 1.3 <= wy <= 3.9)
            if in_corridor:
                if occ[iy, ix] == 0.5: tp += 1
                else: fp_val += 1
    fpr = tp / (tp + fp_val) * 100 if tp + fp_val > 0 else 0

    tr = sum(1 for iy in range(mapper.h) for ix in range(mapper.w)
             if (-0.3 <= mapper.g2w(ix, iy)[0] <= 5.9 and -1.3 <= mapper.g2w(ix, iy)[1] <= 1.3) or
                (3.1 <= mapper.g2w(ix, iy)[0] <= 5.9 and 1.3 <= mapper.g2w(ix, iy)[1] <= 3.9))
    te = sum(1 for iy in range(mapper.h) for ix in range(mapper.w)
             if seen[iy, ix] and (
                (-0.3 <= mapper.g2w(ix, iy)[0] <= 5.9 and -1.3 <= mapper.g2w(ix, iy)[1] <= 1.3) or
                (3.1 <= mapper.g2w(ix, iy)[0] <= 5.9 and 1.3 <= mapper.g2w(ix, iy)[1] <= 3.9)))
    er = te / tr * 100 if tr > 0 else 0

    print(f"\n  Trajectory: {len(traj)} steps, final ({fpos[0]:.2f},{fpos[1]:.2f})")
    print(f"  Frontiers: {len(front_log)}, explored {er:.1f}%, free prec {fpr:.1f}%")
    print(f"  Min clearance: {min_clr:.4f}, RTL dist: {rtl_d:.4f}, collision: {collided}")

    print("\n" + "="*50)
    print("  Pass Criteria")
    print("="*50)
    ok = True
    r1 = er > 50.0; ok &= r1
    print(f"  1. Explored > 50%: {er:.1f}%  {'PASS' if r1 else 'FAIL'}")
    r2 = not collided; ok &= r2
    print(f"  2. No collision: {'PASS' if r2 else 'FAIL'}")
    r3 = min_clr > 0.20; ok &= r3
    print(f"  3. Clearance > 0.20: {min_clr:.4f}  {'PASS' if r3 else 'FAIL'}")
    r4 = len(front_log) > 0; ok &= r4
    print(f"  4. Frontiers: {len(front_log)}  {'PASS' if r4 else 'FAIL'}")
    r5 = rtl_d < 0.05; ok &= r5
    print(f"  5. RTL < 5 cm: {rtl_d:.4f}  {'PASS' if r5 else 'FAIL'}")
    print(f"\n  FRONTIER EXPLORATION {'PASS' if ok else 'FAIL'}")
    return ok

run_frontier_exploration()
