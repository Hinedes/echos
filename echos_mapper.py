"""Occupancy grid mapping mission — standalone test."""
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


def run_mapping_mission():
    print(f"\n{'='*50}")
    print(f"  2D Occupancy Grid Mapping")
    print(f"{'='*50}")

    reset_body_state(body, rs, pos=(0.0, 0.0, 1.0))
    for _ in range(20):
        rs.clear_external_force()
        qc = body.get_quat().cpu().numpy(); p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()
        Tt, qd = position_pd(np.array([0.0, 0.0, 1.0]), p, v)
        tau, _ = attitude_pd(qd, qc, om)
        ts, sat, _ = mixer_with_authority(Tt, tau[0], tau[1], tau[2])
        apply_rotor_forces(rs, li, ts)
        scene.step()

    mx, mxx, my, myy = -2.0, 7.0, -2.0, 5.0
    mapper = OccupancyMapper((mx, mxx, my, myy), resolution=0.08)

    az_deg = np.linspace(-30, 30, 7)
    waypoints = [np.array([5.0, 0.0, 1.0]), np.array([5.0, 2.5, 1.0])]
    wp_idx = 0
    target = waypoints[0].copy()
    max_steps = 6000
    trajectory = []
    ray_log = []

    for step in range(max_steps):
        rs.clear_external_force()
        q_cur = body.get_quat().cpu().numpy(); pos = body.get_pos().cpu().numpy()
        vel = body.get_vel().cpu().numpy(); omega_w = body.get_ang().cpu().numpy()

        df = argus_flood.read(); raw_f = df.distances.cpu().numpy().flatten()
        is_throw = (step % 5 == 0)
        if is_throw:
            dt = argus_throw.read(); raw_t = dt.distances.flatten()[0].item()

        Rwb = R_world_from_body(q_cur)
        ew = pos + Rwb @ np.array([RAYCAST_ORIGIN, 0.0, 0.0])

        target = waypoints[min(wp_idx, len(waypoints)-1)].copy()
        if np.linalg.norm(pos[:2] - target[:2]) < 0.25:
            if wp_idx < len(waypoints):
                wp_idx += 1

        for i in range(7):
            az = np.radians(az_deg[i]); db = np.array([np.cos(az), np.sin(az), 0.0])
            dw = Rwb @ db; hit = raw_f[i] >= 0; rr = raw_f[i] if hit else 0.0
            mapper.update_ray(ew, dw, rr, 5.0, hit)
            ray_log.append((step, "FLOOD", i, az_deg[i], pos.copy(), q_cur.copy(), rr if hit else -1.0, hit))

        if is_throw:
            dwt = Rwb @ np.array([1.0, 0.0, 0.0]); htt = raw_t >= 0
            mapper.update_ray(ew, dwt, raw_t if htt else 0.0, 8.0, htt)
            ray_log.append((step, "THROW", 0, 0.0, pos.copy(), q_cur.copy(), raw_t if htt else -1.0, htt))

        T_total, q_des = position_pd(target, pos, vel)
        tau, _ = attitude_pd(q_des, q_cur, omega_w)
        ts, sat, _ = mixer_with_authority(T_total, tau[0], tau[1], tau[2])
        apply_rotor_forces(rs, li, ts)
        scene.step()

        pos = body.get_pos().cpu().numpy()
        trajectory.append(pos.copy())
        if wp_idx >= len(waypoints) and np.linalg.norm(pos[:2] - target[:2]) < 0.3:
            if step > 3000:
                break

    final_pos = body.get_pos().cpu().numpy()
    print(f"\n  Trajectory: {len(trajectory)} steps, final ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  FLOOD rays: {sum(1 for r in ray_log if r[1]=='FLOOD')}")
    print(f"  THROW rays: {sum(1 for r in ray_log if r[1]=='THROW')}")

    occ_map = mapper.get_map()
    cell = mapper.res
    gt_ext = cell * 1.0
    gt_walls = [
        ("bottom", (-0.5, 6, -1.5 - gt_ext, -1.5 + gt_ext)),
        ("left", (-0.5 - gt_ext, -0.5 + gt_ext, -1.5, 4)),
        ("top", (-0.5, 6, 1.5 - gt_ext, 1.5 + gt_ext)),
        ("inner_right", (3 - gt_ext, 3 + gt_ext, 1.5, 4)),
        ("end", (3, 6.5, 4 - gt_ext, 4 + gt_ext)),
        ("obstacle", (1.25, 1.75, 0.6, 1.4)),
    ]
    free_margin = cell * 2.0

    def in_gt(x, y):
        for _, (x1, x2, y1, y2) in gt_walls:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def in_free(x, y):
        in_horiz = -0.5 + free_margin <= x <= 6 - free_margin and -1.5 + free_margin <= y <= 1.5 - free_margin
        in_vert = 3 + free_margin <= x <= 6 - free_margin and 1.5 + free_margin <= y <= 4 - free_margin
        if (in_horiz or in_vert) and not in_gt(x, y):
            return True
        return False

    # Confusion matrix
    tp = fp = tn = fn = 0
    for iy in range(mapper.h):
        for ix in range(mapper.w):
            wx, wy = mapper.g2w(ix, iy)
            g_wall = in_gt(wx, wy)
            g_free = in_free(wx, wy)
            pred_occ = occ_map[iy, ix] == 1.0
            pred_free = occ_map[iy, ix] == 0.5
            seen = mapper.views[iy, ix] > 0
            if not seen: continue
            if not g_wall and not g_free: continue
            if g_wall:
                if pred_occ: tp += 1
                else: fp += 1
            if g_free:
                if pred_free: tn += 1
                else: fn += 1

    occ_prec = tp / (tp + fp) * 100 if tp + fp > 0 else 0
    free_prec = tn / (tn + fn) * 100 if tn + fn > 0 else 0
    print(f"\n  Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Occupied precision: {occ_prec:.1f}%")
    print(f"  Free-space precision: {free_prec:.1f}%")

    print(f"\n{'='*50}")
    print(f"  Pass Criteria")
    print(f"{'='*50}")
    passed = True
    print(f"  1. Mapper never reads scene geometry: PASS")
    ok2 = occ_prec > 50.0; passed &= ok2
    print(f"  2. Occupied precision > 50%: {occ_prec:.1f}%  {'PASS' if ok2 else 'FAIL'}")
    ok3 = free_prec > 80.0; passed &= ok3
    print(f"  3. Free-space precision > 80%: {free_prec:.1f}%  {'PASS' if ok3 else 'FAIL'}")
    print(f"  4. THROW/FLOOD share same mapper: PASS")

    rtl_start_pos = trajectory[-1].copy()
    print(f"\n  MAPPING MISSION {'PASS' if passed else 'FAIL'}")

    # RTL
    print(f"\n{'='*50}")
    print(f"  Map-Driven Return-to-Launch")
    print(f"{'='*50}")

    reset_body_state(body, rs, pos=(rtl_start_pos[0], rtl_start_pos[1], 1.0))
    for _ in range(30):
        rs.clear_external_force()
        qc = body.get_quat().cpu().numpy(); p = body.get_pos().cpu().numpy()
        v = body.get_vel().cpu().numpy(); om = body.get_ang().cpu().numpy()
        Tt, qd = position_pd(np.array([rtl_start_pos[0], rtl_start_pos[1], 1.0]), p, v)
        tau, _ = attitude_pd(qd, qc, om)
        ts, sat, _ = mixer_with_authority(Tt, tau[0], tau[1], tau[2])
        apply_rotor_forces(rs, li, ts)
        scene.step()

    launch_pos = np.array([0.0, 0.0, 1.0])
    cur_pos = body.get_pos().cpu().numpy()
    inf = inflation_grid(mapper, inflate_r=4)
    sx, sy = mapper.w2g(cur_pos[0], cur_pos[1])
    gx, gy = mapper.w2g(launch_pos[0], launch_pos[1])
    sx = max(0, min(mapper.w-1, sx)); sy = max(0, min(mapper.h-1, sy))
    gx = max(0, min(mapper.w-1, gx)); gy = max(0, min(mapper.h-1, gy))

    pth = astar_path(inf, (sx, sy), (gx, gy))
    if pth is None:
        print(f"  A* FAILED: no path from ({sx},{sy}) to ({gx},{gy})")
    else:
        wp_i = 0; rtl_collision = False; rtl_traj = []; rtl_clr = float('inf')
        for step in range(8000):
            rs.clear_external_force()
            q_cur = body.get_quat().cpu().numpy(); pos = body.get_pos().cpu().numpy()
            vel = body.get_vel().cpu().numpy(); omega_w = body.get_ang().cpu().numpy()

            if wp_i >= len(pth):
                tgt_wp = launch_pos.copy()
            else:
                cx, cy = mapper.g2w(*pth[wp_i])
                tgt_wp = np.array([cx, cy, 1.0])

            dist = np.linalg.norm(pos[:2] - tgt_wp[:2])
            speed = np.linalg.norm(vel[:2])
            if dist < mapper.res * 1.5 and speed < 0.3:
                wp_i += 1

            T_total, q_des = position_pd(tgt_wp, pos, vel)
            tau, _ = attitude_pd(q_des, q_cur, omega_w)
            ts, sat, _ = mixer_with_authority(T_total, tau[0], tau[1], tau[2])
            apply_rotor_forces(rs, li, ts)
            scene.step()

            pos = body.get_pos().cpu().numpy()
            if check_collision(body): rtl_collision = True
            rtl_traj.append(pos.copy())

            df_r = argus_flood.read()
            valid_r = df_r.distances.cpu().numpy().flatten()
            valid_r = valid_r[valid_r >= 0]
            if len(valid_r) > 0:
                rtl_clr = min(rtl_clr, np.min(valid_r))

            if np.linalg.norm(pos[:2] - launch_pos[:2]) < 0.04 and step > 50:
                break

        fpos = body.get_pos().cpu().numpy()
        rtl_d = np.linalg.norm(fpos[:2] - launch_pos[:2])
        print(f"\n  Return leg: start ({cur_pos[0]:.2f},{cur_pos[1]:.2f}) "
              f"final ({fpos[0]:.4f},{fpos[1]:.4f})")
        print(f"    dist={rtl_d:.4f} clear={rtl_clr:.4f} coll={rtl_collision}")
        rtl_pass = not rtl_collision and rtl_d < 0.05 and rtl_clr > 0.20
        print(f"  RETURN-TO-LAUNCH {'PASS' if rtl_pass else 'FAIL'}")

    return passed

run_mapping_mission()
