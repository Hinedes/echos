import heapq

def astar_path(grid, inflated, start, goal):
    h, w = grid.shape
    def free(ix, iy):
        return 0 <= ix < w and 0 <= iy < h and inflated[iy, ix] == 0
    sx, sy = start; gx, gy = goal
    if not free(sx, sy) or not free(gx, gy): return None
    opens = [(0, (sx, sy))]; came = {}; g_c = {(sx, sy): 0}
    while opens:
        _, cur = heapq.heappop(opens)
        if cur == (gx, gy):
            p = []; c = cur
            while c in came: p.append(c); c = came[c]
            p.append((sx, sy)); p.reverse(); return p
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not free(nx, ny): continue
            if dx != 0 and dy != 0:
                if not free(cur[0]+dx, cur[1]): continue
                if not free(cur[0], cur[1]+dy): continue
            ng = g_c[cur] + (1.414 if dx != 0 and dy != 0 else 1.0)
            if (nx, ny) not in g_c or ng < g_c[(nx, ny)]:
                came[(nx, ny)] = cur; g_c[(nx, ny)] = ng
                heapq.heappush(opens, (ng + abs(nx-gx) + abs(ny-gy), (nx, ny)))
    return None

def run_frontier_exploration():
    print("\n" + "="*50)
    print("  Autonomous Frontier Exploration")
    print("="*50)
    print("  L-corridor, FLOOD+THROW, no predefined route")

    reset_body(pos=(0.0, 0.0, 1.0))
    for _ in range(20):
        rigid_solver.clear_external_force()
        qc = body.get_quat().cpu().numpy(); p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()
        Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
        apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, qc, om)), 0.0, None))
        scene.step()

    mx, mxx, my, myy = -2.0, 7.0, -2.0, 5.0
    mapper = OccupancyMapper((mx, mxx, my, myy), 0.08)
    eps = 0.004; az_deg = np.linspace(-30, 30, 7); az_cos = np.cos(np.radians(az_deg))
    launch = np.array([0.0, 0.0, 1.0])
    max_steps = 15000; replan_iv = 120
    state = "INITIAL_FLY"
    cur_path = []; wp_i = 0
    traj = []; scan_log = []; front_log = []; dec = []
    min_clr = float('inf'); col = False; best = None
    observe_steps = 0; yaw_aligned = False

    for step in range(max_steps):
        rigid_solver.clear_external_force()
        q_cur = body.get_quat().cpu().numpy(); pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()

        df = argus_flood.read(); raw_f = df.distances.cpu().numpy().flatten()
        r_f = raw_f + eps * az_cos
        vf = r_f[r_f >= 0]
        if len(vf) > 0: min_clr = min(min_clr, np.min(vf))
        is_thr = (step % 5 == 0)
        if is_thr:
            dt = argus_throw.read(); raw_t = dt.distances.flatten()[0].item()
        R_bw = rot_world_to_body(q_cur).T
        ew = pos + R_bw @ np.array([emit_off[0], 0.0, 0.0])
        for i in range(7):
            az = np.radians(az_deg[i]); db = np.array([np.cos(az), np.sin(az), 0.0])
            dw = R_bw @ db; hit = raw_f[i] >= 0; rr = raw_f[i] if hit else 0.0
            mapper.update_ray(ew, dw, rr, 5.0, hit, eps)
        if is_thr:
            dwt = R_bw @ np.array([1.0, 0.0, 0.0]); htt = raw_t >= 0
            mapper.update_ray(ew, dwt, raw_t if htt else 0.0, 8.0, htt, eps)

        if step < 300:
            state = "INITIAL_FLY"
        elif state == "INITIAL_FLY":
            state = "EXPLORE"
        if state == "INITIAL_FLY":
            tgt = np.array([pos[0]+2.0, 0.0, 1.0])
            Tt, qd = position_pd(tgt, pos, vel)
            apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))
            scene.step()
            pos = body.get_pos().cpu().numpy()
            if pos[2] <= 0.01: col = True
            traj.append(pos.copy()); scan_log.append((step, r_f.copy(), state))
            continue

        if state == "EXPLORE" and step % replan_iv == 0:
            occ = mapper.get_map(); h, w = occ.shape
            inf = np.ones_like(occ)
            for iy in range(h):
                for ix in range(w):
                    if occ[iy, ix] == 0.5: inf[iy, ix] = 0
            for iy in range(h):
                for ix in range(w):
                    if occ[iy, ix] == 1.0:
                        for dy in range(-4, 5):
                            for dx in range(-4, 5):
                                nx, ny = ix+dx, iy+dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    if occ[ny, nx] == 0.5: continue
                                    inf[ny, nx] = 1.0
            for tp in traj:
                tix, tiy = mapper.w2g(tp[0], tp[1])
                for ddx in range(-2, 3):
                    for ddy in range(-2, 3):
                        nx, ny = tix+ddx, tiy+ddy
                        if 0 <= nx < w and 0 <= ny < h: inf[ny, nx] = 0

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
                    if not ns: lab[iy, ix] = cl; cl += 1
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

            cld = []
            sx, sy = mapper.w2g(pos[0], pos[1])
            best = None; bs = -1; bk = None

            for l, cells in cs.items():
                if len(cells) < 5: continue

                # 1. Find UNKNOWN cells adjacent to this cluster's frontier cells
                adj_unk = set()
                for (ix, iy) in cells:
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = ix+dx, iy+dy
                        if 0 <= nx < w and 0 <= ny < h and occ[ny, nx] == 0.0:
                            adj_unk.add((nx, ny))
                if len(adj_unk) == 0: continue

                # 2. Centroid of adjacent unknown cells
                uxs = [p[0] for p in adj_unk]; uys = [p[1] for p in adj_unk]
                uc_x = sum(uxs)/len(uxs); uc_y = sum(uys)/len(uys)

                # 3. Among cluster's frontier cells, find nearest to unknown centroid
                best_d = 1e9; gx, gy = cells[0]
                for (ix, iy) in cells:
                    if not (occ[iy, ix] == 0.5 and inf[iy, ix] == 0): continue
                    d = abs(ix - uc_x) + abs(iy - uc_y)
                    if d < best_d: best_d = d; gx, gy = ix, iy

                # 4. A* to chosen frontier cell
                pth = astar_path(occ, inf, (sx, sy), (gx, gy))
                if pth is None: continue

                cx, cy = mapper.g2w(gx, gy)
                gn = len(cells); ct = len(pth) * mapper.res
                sc = gn / (ct + 0.01)
                cl = {"cx": cx, "cy": cy, "cix": gx, "ciy": gy, "n": gn, "cells": cells,
                      "ucx": uc_x, "ucy": uc_y, "adj_unk_n": len(adj_unk)}
                ky = (sc, gx, gy)
                if best is None or sc > bs or (abs(sc-bs) < 1e-6 and ky < bk):
                    best = cl; bs = sc; bk = ky; cur_path = pth
                cld.append(cl)

            cld.sort(key=lambda c: -c['n'])
            cld = cld[:20]

            # Print all surviving clusters
            if len(cld) > 0:
                print(f"  step {step}: {len(cld)} clusters evaluated:")
                for ci, cl in enumerate(cld):
                    pth = astar_path(occ, inf, (sx, sy), (cl["cix"], cl["ciy"]))
                    pc = len(pth) * mapper.res if pth else float('inf')
                    sc = cl["n"] / (pc + 0.01) if pth else 0
                    is_best = (best is not None and cl["cix"] == best["cix"] and cl["ciy"] == best["ciy"])
                    uc_wx = mapper.x_min + (cl["ucx"] + 0.5) * mapper.res
                    uc_wy = mapper.y_min + (cl["ucy"] + 0.5) * mapper.res
                    print(f"    #{ci}: goal=({cl['cx']:.2f},{cl['cy']:.2f}) n={cl['n']} "
                          f"unk_cent=({uc_wx:.2f},{uc_wy:.2f}) cost={pc:.2f} score={sc:.3f}"
                          f"{' [SELECTED]' if is_best else ''}")
            else:
                print(f"  step {step}: no clusters survive")

            if best is None:
                dec.append((step, "NO_FRONTIER", pos[0], pos[1]))
                state = "RTL"
                print(f"  step {step}: no reachable frontier -> RTL")
            else:
                uc_wx = mapper.x_min + (best["ucx"] + 0.5) * mapper.res
                uc_wy = mapper.y_min + (best["ucy"] + 0.5) * mapper.res
                best["observe_heading"] = np.arctan2(uc_wy - best["cy"], uc_wx - best["cx"])
                best["uk_world"] = (uc_wx, uc_wy)
                dec.append((step, "FRONTIER", best["cx"], best["cy"]))
                front_log.append({"step": step, "cx": best["cx"], "cy": best["cy"], "n": best["n"], "score": bs,
                                  "observe_heading": best["observe_heading"]})
                state = "FLY_TO_FRONTIER"; wp_i = 0
                if len(front_log) <= 10:
                    uc_wx = mapper.x_min + (best["ucx"] + 0.5) * mapper.res
                    uc_wy = mapper.y_min + (best["ucy"] + 0.5) * mapper.res
                    print(f"  step {step}: frontier #{len(front_log)} ({best['cx']:.2f},{best['cy']:.2f}) n={best['n']} "
                          f"unk_cent=({uc_wx:.2f},{uc_wy:.2f}) score={bs:.3f}")
                pass  # continue running

        if state == "FLY_TO_FRONTIER":
            if wp_i >= len(cur_path):
                state = "HOLD_AND_OBSERVE"
                observe_steps = 0
                yaw_aligned = False
                if best is not None:
                    print(f"  step {step}: reached frontier, observe heading={np.degrees(best['observe_heading']):.1f}°")
            else:
                cx, cy = mapper.g2w(*cur_path[wp_i])
                tgt = np.array([cx, cy, 1.0])
                if np.linalg.norm(pos[:2] - tgt[:2]) < 0.25: wp_i += 1
                Tt, qd = position_pd(tgt, pos, vel)
                apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))
        elif state == "HOLD_AND_OBSERVE":
            if not yaw_aligned and best is not None and observe_steps < 200:
                psi_des = -best["observe_heading"]
                Tt, qd = position_pd(np.array([pos[0], pos[1], 1.0]), pos, vel, psi_des)
                apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))
                # Check alignment: body X axis dot unknown direction
                q_cur_a = body.get_quat().cpu().numpy()
                body_x = rot_world_to_body(q_cur_a).T @ np.array([1.0, 0.0, 0.0])
                uk_x, uk_y = best["uk_world"]
                unk_dir = np.array([uk_x - best["cx"], uk_y - best["cy"], 0.0])
                unk_dir /= np.linalg.norm(unk_dir)
                dot_val = body_x @ unk_dir
                if observe_steps == 0:
                    print(f"    yaw aligning... heading={np.degrees(best['observe_heading']):.1f} dot={dot_val:.4f}")
                if dot_val > 0.99:
                    yaw_aligned = True
                    _, _, a_yaw = quat_to_euler(body.get_quat().cpu().numpy())
                    print(f"    aligned: yaw={a_yaw:.2f}° dot={dot_val:.4f}")
            elif yaw_aligned:
                observe_steps += 1
                # Hold position and scan (FLOOD reading happens at top of loop)
                Tt, qd = position_pd(np.array([pos[0], pos[1], 1.0]), pos, vel)
                apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))
                if observe_steps >= 30:
                    uc_before = np.sum(mapper.views > 0)
                    state = "EXPLORE"
                    uc_after = np.sum(mapper.views > 0)
                    print(f"    scan complete: viewed cells {uc_before} -> {uc_after}")
            else:
                Tt, qd = position_pd(np.array([pos[0], pos[1], 1.0]), pos, vel)
                apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))

        elif state == "EXPLORE":
            Tt, qd = position_pd(np.array([pos[0], pos[1], 1.0]), pos, vel)
            apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))

        scene.step()
        pos = body.get_pos().cpu().numpy()
        if pos[2] <= 0.01: col = True
        traj.append(pos.copy()); scan_log.append((step, r_f.copy(), state))

    # RTL
    occ = mapper.get_map(); h, w = occ.shape
    inf = np.ones_like(occ)
    for iy in range(h):
        for ix in range(w):
            if occ[iy, ix] == 0.5: inf[iy, ix] = 0
    for iy in range(h):
        for ix in range(w):
            if occ[iy, ix] == 1.0:
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        nx, ny = ix+dx, iy+dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if occ[ny, nx] == 0.5: continue
                            inf[ny, nx] = 1.0
    for tp in traj:
        tix, tiy = mapper.w2g(tp[0], tp[1])
        for ddx in range(-2, 3):
            for ddy in range(-2, 3):
                nx, ny = tix+ddx, tiy+ddy
                if 0 <= nx < w and 0 <= ny < h: inf[ny, nx] = 0
    # Force start and goal traversable
    sxi, syi = mapper.w2g(pos[0], pos[1])
    gxi, gyi = mapper.w2g(launch[0], launch[1])
    for ddx in range(-2, 3):
        for ddy in range(-2, 3):
            nx, ny = sxi+ddx, syi+ddy
            if 0 <= nx < w and 0 <= ny < h: inf[ny, nx] = 0
            nx, ny = gxi+ddx, gyi+ddy
            if 0 <= nx < w and 0 <= ny < h: inf[ny, nx] = 0

    sx, sy = sxi, syi; gx, gy = gxi, gyi
    pth = astar_path(occ, inf, (sx, sy), (gx, gy))
    if pth is None:
        print(f"  RTL FAILED: no path to launch")
    else:
        print(f"  RTL path: {len(pth)} cells")
        for s2 in range(8000):
            rigid_solver.clear_external_force()
            qc = body.get_quat().cpu().numpy(); p2 = body.get_pos().cpu().numpy()
            v2 = body.get_vel().cpu().numpy(); om2 = body.get_ang().cpu().numpy()
            wi2 = min(s2 // 3, len(pth)-1)
            cx2, cy2 = mapper.g2w(*pth[wi2])
            tgt2 = np.array([cx2, cy2, 1.0])
            if np.linalg.norm(p2[:2] - launch[:2]) < 0.04:
                if s2 > 50: break
            Tt, qd = position_pd(tgt2, p2, v2)
            apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, qc, om2)), 0.0, None))
            scene.step()

    fpos = body.get_pos().cpu().numpy()
    rtl_d = np.linalg.norm(fpos[:2] - launch[:2])
    of = mapper.get_map(); seen = mapper.views > 0

    tp = 0; fp_val = 0
    for iy in range(mapper.h):
        for ix in range(mapper.w):
            wx, wy = mapper.g2w(ix, iy)
            if not seen[iy, ix]: continue
            # Approximate free space: inside corridor bounds, not on walls
            in_corridor = (-0.3 <= wx <= 5.9 and -1.3 <= wy <= 1.3) or \
                          (3.1 <= wx <= 5.9 and 1.3 <= wy <= 3.9)
            if in_corridor:
                if of[iy, ix] == 0.5: tp += 1
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
    print(f"  Min clearance: {min_clr:.4f}, RTL dist: {rtl_d:.4f}, coll: {col}")

    print("\n" + "="*50)
    print("  Pass Criteria")
    print("="*50)
    ok = True
    r1 = er > 50.0; ok &= r1
    print(f"  1. Explored > 50%: {er:.1f}%  {'PASS' if r1 else 'FAIL'}")
    r2 = not col; ok &= r2
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
