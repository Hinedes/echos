"""Sweep over yaw/position/beam combos to find raycaster misses from +X."""
import genesis as gs
import numpy as np

gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
body = scene.add_entity(gs.morphs.Box(size=(0.165,0.165,0.0545), pos=(0,0,1), fixed=False))
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True))
sensor = scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(fov=(60,0), n_points=(7,1)),
    entity_idx=body.idx, pos_offset=(0.0865,0,0), max_range=5))
scene.build(); body.set_mass(0.225)

fails = []; total = 0
az_d = np.linspace(-30, 30, 7)
for yaw_deg in range(0, 360, 30):
    yaw = np.radians(yaw_deg)
    R = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    for px in np.arange(2.5, 4.01, 0.5):
        for py in np.arange(1.5, 4.01, 0.5):
            body.set_pos(np.array([px, py, 1.0]))
            body.set_quat(np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]))
            scene.step(); scene.step(); scene.step()
            d = sensor.read(); rf = d.distances.cpu().numpy().flatten()
            for i,az in enumerate(np.radians(az_d)):
                db = np.array([np.cos(az),np.sin(az),0]); dw = R @ db
                if dw[0] < 0:  # heading toward wall
                    total += 1
                    # Sensor world pos = body_pos + R @ (0.0865,0,0)
                    cos_yaw = np.cos(yaw)
                    sx = px + 0.0865 * cos_yaw
                    wall_t = (3.005 - sx) / dw[0]
                    if 0 < wall_t < 5 and rf[i] < 0:
                        fails.append((yaw_deg, px, py, i, dw[0], dw[1], wall_t, rf[i]))

print("Total rays toward wall: %d, unexpected MISS: %d" % (total, len(fails)))
for f in fails[:20]:
    yaw_deg, px, py, i, dx, dy, wall_t, raw = f
    print("  YAW=%d P=(%.1f,%.1f) az=%d dir=(%.3f,%.3f) wall_t=%.2f raw=%.4f" % (yaw_deg, px, py, i, dx, dy, wall_t, raw))
gs.destroy()
