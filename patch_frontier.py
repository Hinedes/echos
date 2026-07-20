import sys

with open(sys.argv[1], 'r') as f:
    src = f.read()

# Find run_mapping_mission and replace it with frontier exploration
marker = 'def run_mapping_mission():'
new_func = '''
import heapq

def astar_path(grid, inflated, start, goal):
    h, w = grid.shape
    def free(ix, iy):
        return 0 <= ix < w and 0 <= iy < h and inflated[iy, ix] == 0
    sx, sy = start; gx, gy = goal
    if not free(sx, sy) or not free(gx, gy):
        return None
    opens = [(0, (sx, sy))]
    came = {}; g_c = {(sx, sy): 0}
    while opens:
        _, cur = heapq.heappop(opens)
        if cur == (gx, gy):
            p = []; c = cur
            while c in came:
                p.append(c); c = came[c]
            p.append((sx, sy)); p.reverse(); return p
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = cur[0] + dx, cur[1] + dy
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
    print(f"\\n{\\"=\\"*50}")
    print(f"  Autonomous Frontier Exploration")
    print(f"{\\"=\\"*50}")
    print(f"  L-corridor: horizontal -> vertical")
    print(f"  FLOOD continuous, THROW occasional")
    print(f"  No predefined route -- exploring from map")

    reset_body(pos=(0.0, 0.0, 1.0))
    for _ in range(20):
        rigid_solver.clear_external_force()
        qc = body.get_quat().cpu().numpy(); p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()
        Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
        apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, qc, om)), 0.0, None))
        scene.step()

    m_xmin, m_xmax, m_ymin, m_ymax = -2.0, 7.0, -2.0, 5.0
    mapper = OccupancyMapper((m_xmin, m_xmax, m_ymin, m_ymax), resolution=0.08)
    eps = 0.004; az_deg = np.linspace(-30, 30, 7); az_cos = np.cos(np.radians(az_deg))
    launch = np.array([0.0, 0.0, 1.0])
    max_steps = 15000; replan_interval = 120
    state = "INITIAL_FLY"
    current_path = []; wp_idx = 0
    trajectory = []; scan_log = []; frontier_log = []; decisions = []
    min_clearance = float(\'inf\'); collision = False

    for step in range(max_steps):
        rigid_solver.clear_external_force()
        q_cur = body.get_quat().cpu().numpy(); pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()

        df = argus_flood.read(); raw_f = df.distances.cpu().numpy().flatten()
        ranges_f = raw_f + eps * az_cos
        valid_f = ranges_f[ranges_f >= 0]
        if len(valid_f) > 0: min_clearance = min(min_clearance, np.min(valid_f))

        is_throw = (step % 5 == 0)
        if is_throw:
            dt = argus_throw.read(); raw_t = dt.distances.flatten()[0].item()

        R_bw = rot_world_to_body(q_cur).T
        emitter_w = pos + R_bw @ np.array([emit_off[0], 0.0, 0.0])
        for i in range(7):
            az = np.radians(az_deg[i]); db = np.array([np.cos(az), np.sin(az), 0.0])
            dw = R_bw @ db; hit = raw_f[i] >= 0; rr = raw_f[i] if hit else 0.0
            mapper.update_ray(emitter_w, dw, rr, 5.0, hit, eps)
        if is_throw:
            dwt = R_bw @ np.array([1.0, 0.0, 0.0]); hitt = raw_t >= 0
            mapper.update_ray(emitter_w, dwt, raw_t if hitt else 0.0, 8.0, hitt, eps)

        # Fly forward for first 300 steps to build map
        if step < 300:
            state = "INITIAL_FLY"
        if state == "INITIAL_FLY":
            target = np.array([pos[0] + 2.0, 0.0, 1.0])
            Tt, qd = position_pd(target, pos, vel)
            apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))
            scene.step()
            pos = body.get_pos().cpu().numpy()
            if pos[2] <= 0.01: collision = True
            trajectory.append(pos.copy())
            scan_log.append((step, ranges_f.copy(), state))
            continue

        # Frontier exploration (every replan_interval steps)
        if state == "EXPLORE" and step % replan_interval == 0:
            occ = mapper.get_map()
            h, w = occ.shape
            inflate_r = 4
            inflated = np.ones_like(occ)
            for iy in range(h):
                for ix in range(w):
                    if occ[iy, ix] == 0.5: inflated[iy, ix] = 0
            for iy in range(h):
                for ix in range(w):
                    if occ[iy, ix] == 1.0:
                        for dy in range(-inflate_r, inflate_r + 1):
                            for dx in range(-inflate_r, inflate_r + 1):
                                nx, ny = ix + dx, iy + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    if occ[ny, nx] == 0.5: continue
                                    inflated[ny, nx] = 1.0
            for tp in trajectory:
                tix, tiy = mapper.w2g(tp[0], tp[1])
                for ddx in range(-2, 3):
                    for ddy in range(-2, 3):
                        nx, ny = tix + ddx, tiy + ddy
                        if 0 <= nx < w and 0 <= ny < h: inflated[ny, nx] = 0

            # Frontier detection: free cell adjacent to unknown
            fg = np.zeros((h, w), dtype=np.int32)
            for iy in range(1, h - 1):
                for ix in range(1, w - 1):
                    if occ[iy, ix] == 0.5 and inflated[iy, ix] == 0:
                        if (occ[iy-1, ix] == 0.0 or occ[iy+1, ix] == 0.0 or
                            occ[iy, ix-1] == 0.0 or occ[iy, ix+1] == 0.0):
                            fg[iy, ix] = 1

            # Two-pass connected component labeling
            label = np.zeros((h, w), dtype=np.int32)
            cur_l = 1; equiv = {}
            for iy in range(1, h - 1):
                for ix in range(1, w - 1):
                    if fg[iy, ix] == 0: continue
                    up = label[iy-1, ix]; left = label[iy, ix-1]
                    ul = label[iy-1, ix-1]; ur = label[iy-1, ix+1]
                    nbs = [l for l in [up, left, ul, ur] if l > 0]
                    if not nbs:
                        label[iy, ix] = cur_l; cur_l += 1
                    else:
                        m = min(nbs); label[iy, ix] = m
                        for n in nbs:
                            if n != m: equiv[n] = m
            for iy in range(1, h - 1):
                for ix in range(1, w - 1):
                    l = label[iy, ix]
                    if l > 0:
                        while l in equiv: l = equiv[l]
                        label[iy, ix] = l

            clust_stats = {}
            for iy in range(1, h - 1):
                for ix in range(1, w - 1):
                    l = label[iy, ix]
                    if l > 0:
                        if l not in clust_stats: clust_stats[l] = []
                        clust_stats[l].append((ix, iy))

            clustered = []
            for l, cells in clust_stats.items():
                if len(cells) < 5: continue
                xs = [p[0] for p in cells]; ys = [p[1] for p in cells]
                cix = int(round(sum(xs) / len(xs))); ciy = int(round(sum(ys) / len(ys)))
                cix = max(1, min(w - 2, cix)); ciy = max(1, min(h - 2, ciy))
                best_d = 999; gx, gy = cix, ciy
                for dd in range(-3, 4):
                    for ee in range(-3, 4):
                        nx, ny = cix + dd, ciy + ee
                        if 0 <= nx < w and 0 <= ny < h and occ[ny, nx] == 0.5 and inflated[ny, nx] == 0:
                            d = abs(dd) + abs(ee)
                            if d < best_d: best_d = d; gx, gy = nx, ny
                cx, cy = mapper.g2w(gx, gy)
                clustered.append({"cx": cx, "cy": cy, "cix": gx, "ciy": gy, "n": len(cells)})

            clustered.sort(key=lambda c: -c['n'])
            clustered = clustered[:20]

            sx, sy = mapper.w2g(pos[0], pos[1])
            best = None; best_score = -1; best_key = None
            for cl in clustered:
                if cl["cx"] < pos[0] + 0.3: continue
                if not (0 <= cl["cix"] < w and 0 <= cl["ciy"] < h): continue
                if not (inflated[cl["ciy"], cl["cix"]] == 0): continue
                if not (occ[cl["ciy"], cl["cix"]] == 0.5): continue
                path = astar_path(occ, inflated, (sx, sy), (cl["cix"], cl["ciy"]))
                if path is None: continue
                gain = cl["n"]; cost = len(path) * mapper.res
                score = gain / (cost + 0.01)
                key = (score, cl["cix"], cl["ciy"])
                if best is None or score > best_score or (abs(score - best_score) < 1e-6 and key < best_key):
                    best = cl; best_score = score; best_key = key; current_path = path

            if best is None:
                decisions.append((step, "NO_FRONTIER", pos[0], pos[1]))
                state = "RTL"
                print(f"  step {step}: no reachable frontier -> RTL")
            else:
                decisions.append((step, "FRONTIER", best["cx"], best["cy"]))
                frontier_log.append({"step": step, "cx": best["cx"], "cy": best["cy"], "n": best["n"], "score": best_score})
                state = "FLY_TO_FRONTIER"; wp_idx = 0
                if len(frontier_log) <= 10:
                    print(f"  step {step}: frontier #{len(frontier_log)} ({best[\"cx\"]:.2f},{best[\"cy\"]:.2f}) {best[\"n\"]} cells score={best_score:.3f}")

        # Execute waypoint path
        if state == "FLY_TO_FRONTIER":
            if wp_idx >= len(current_path):
                state = "EXPLORE"
            else:
                cx, cy = mapper.g2w(*current_path[wp_idx])
                target = np.array([cx, cy, 1.0])
                if np.linalg.norm(pos[:2] - target[:2]) < 0.25:
                    wp_idx += 1
                Tt, qd = position_pd(target, pos, vel)
                tau = attitude_pd(qd, q_cur, om)
                apply_rotor_forces(np.clip(mixer(Tt, *tau), 0.0, None))
        elif state == "EXPLORE":
            Tt, qd = position_pd(np.array([pos[0], pos[1], 1.0]), pos, vel)
            apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, q_cur, om)), 0.0, None))

        scene.step()
        pos = body.get_pos().cpu().numpy()
        if pos[2] <= 0.01: collision = True
        trajectory.append(pos.copy())
        scan_log.append((step, ranges_f.copy(), state))

    # RTL
    occ = mapper.get_map(); h, w = occ.shape
    inflate_r = 4; inflated = np.ones_like(occ)
    for iy in range(h):
        for ix in range(w):
            if occ[iy, ix] == 0.5: inflated[iy, ix] = 0
    for iy in range(h):
        for ix in range(w):
            if occ[iy, ix] == 1.0:
                for dy in range(-inflate_r, inflate_r + 1):
                    for dx in range(-inflate_r, inflate_r + 1):
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if occ[ny, nx] == 0.5: continue
                            inflated[ny, nx] = 1.0
    for tp in trajectory:
        tix, tiy = mapper.w2g(tp[0], tp[1])
        for ddx in range(-2, 3):
            for ddy in range(-2, 3):
                nx, ny = tix + ddx, tiy + ddy
                if 0 <= nx < w and 0 <= ny < h: inflated[ny, nx] = 0

    sx, sy = mapper.w2g(pos[0], pos[1]); gx, gy = mapper.w2g(launch[0], launch[1])
    path = astar_path(occ, inflated, (sx, sy), (gx, gy))
    if path is None:
        print(f"  RTL FAILED: no path to launch")
    else:
        print(f"  RTL path: {len(path)} cells")
        for step2 in range(8000):
            rigid_solver.clear_external_force()
            qc = body.get_quat().cpu().numpy(); p = body.get_pos().cpu().numpy()
            v = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()
            wi = min(step2 // 3, len(path) - 1)
            cx, cy = mapper.g2w(*path[wi])
            target = np.array([cx, cy, 1.0])
            if np.linalg.norm(p[:2] - launch[:2]) < 0.04:
                if step2 > 50: break
            Tt, qd = position_pd(target, p, v)
            apply_rotor_forces(np.clip(mixer(Tt, *attitude_pd(qd, qc, om)), 0.0, None))
            scene.step()

    final_pos = body.get_pos().cpu().numpy()
    rtl_dist = np.linalg.norm(final_pos[:2] - launch[:2])
    occ_final = mapper.get_map(); seen = mapper.views > 0

    # Evaluate
    cell = mapper.res; gt_ext = cell * 1.0
    gt_walls = [
        ("bottom", (-0.5, 6, -1.5 - gt_ext, -1.5 + gt_ext)),
        ("left", (-0.5 - gt_ext, -0.5 + gt_ext, -1.5, 4)),
        ("top", (-0.5, 3.5, 1.5 - gt_ext, 1.5 + gt_ext)),
        ("inner_right", (3 - gt_ext, 3 + gt_ext, 1.5, 4)),
        ("end", (3, 6.5, 4 - gt_ext, 4 + gt_ext)),
        ("obstacle", (1.25, 1.75, 0.6, 1.4)),
    ]
    free_margin = cell * 2.0
    def in_gt(x, y):
        for _, (x1, x2, y1, y2) in gt_walls:
            if x1 <= x <= x2 and y1 <= y <= y2: return True
        return False
    def in_free(x, y):
        ih = -0.5 + free_margin <= x <= 6 - free_margin and -1.5 + free_margin <= y <= 1.5 - free_margin
        iv = 3 + free_margin <= x <= 6 - free_margin and 1.5 + free_margin <= y <= 4 - free_margin
        return (ih or iv) and not in_gt(x, y)

    tp = fp = 0
    for iy in range(mapper.h):
        for ix in range(mapper.w):
            wx, wy = mapper.g2w(ix, iy)
            if not seen[iy, ix]: continue
            g_free = in_free(wx, wy)
            is_free = occ_final[iy, ix] == 0.5
            if g_free and is_free: tp += 1
            elif not g_free and is_free: fp += 1
    free_prec = tp / (tp + fp) * 100 if tp + fp > 0 else 0

    total_reachable = sum(1 for iy in range(mapper.h) for ix in range(mapper.w)
                          if in_free(*mapper.g2w(ix, iy)))
    total_explored = sum(1 for iy in range(mapper.h) for ix in range(mapper.w)
                         if in_free(*mapper.g2w(ix, iy)) and seen[iy, ix])
    explore_ratio = total_explored / total_reachable * 100 if total_reachable > 0 else 0

    print(f"\\n  Trajectory: {len(trajectory)} steps, final ({final_pos[0]:.2f},{final_pos[1]:.2f})")
    print(f"  Frontiers selected: {len(frontier_log)}")
    print(f"  Free-space explored: {total_explored}/{total_reachable} = {explore_ratio:.1f}%")
    print(f"  Free-space precision: {free_prec:.1f}%")
    print(f"  Min clearance: {min_clearance:.4f} m")
    print(f"  RTL dist: {rtl_dist:.4f} m")
    print(f"  Collision: {collision}")

    print(f"\\n{\\"=\\"*50}")
    print(f"  Pass Criteria")
    print(f"{\\"=\\"*50}")
    passed = True
    ok1 = explore_ratio > 50.0; passed &= ok1
    print(f"  1. Explored > 50%: {explore_ratio:.1f}%  {\\"PASS\\" if ok1 else \\"FAIL\\"}")
    ok2 = not collision; passed &= ok2
    print(f"  2. No collision: {\\"PASS\\" if ok2 else \\"FAIL\\"}")
    ok3 = min_clearance > 0.20; passed &= ok3
    print(f"  3. Clearance > 0.20 m: {min_clearance:.4f}  {\\"PASS\\" if ok3 else \\"FAIL\\"}")
    ok4 = len(frontier_log) > 0; passed &= ok4
    print(f"  4. Frontiers selected: {len(frontier_log)}  {\\"PASS\\" if ok4 else \\"FAIL\\"}")
    ok5 = rtl_dist < 0.05; passed &= ok5
    print(f"  5. RTL dist < 5 cm: {rtl_dist:.4f}  {\\"PASS\\" if ok5 else \\"FAIL\\"}")
    print(f"\\n  FRONTIER EXPLORATION {\\"PASS\\" if passed else \\"FAIL\\"}")
    return passed

run_frontier_exploration()
'''

if marker in src:
    # Replace everything from the marker to the end
    idx = src.index(marker)
    # Find the end of the file (last call or end of string)
    src = src[:idx] + new_func
    with open(sys.argv[1], 'w') as f:
        f.write(src)
    print(f"Replaced run_mapping_mission with run_frontier_exploration")
else:
    print(f"ERROR: could not find '{marker}' in file")
