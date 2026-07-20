"""Find which corridor entity causes the raycaster miss at step 1652 az=6."""
import genesis as gs
import numpy as np

gs.init(backend=gs.amdgpu)

def run_test(desc, extra_walls):
    scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
    body = scene.add_entity(gs.morphs.Box(size=(0.165,0.165,0.0545), pos=(0,0,1), fixed=False))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True))
    for wall in extra_walls:
        scene.add_entity(wall)
    sensor = scene.add_sensor(gs.sensors.Raycaster(
        pattern=gs.sensors.SphericalPattern(fov=(60.0,0.0), n_points=(7,1)),
        entity_idx=body.idx, pos_offset=(0.0865, 0.0, 0.0),
        euler_offset=(0.0, 0.0, 0.0), max_range=5.0, no_hit_value=-1.0))
    scene.build()
    body.set_mass(0.225)

    yaw = np.radians(75)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0])
    body_pos = np.array([3.077, 2.794, 1.0]) - 0.0865 * fwd
    body.set_pos(body_pos)
    body.set_quat(np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))
    for _ in range(10): scene.step()

    d = sensor.read()
    rf = d.distances.cpu().numpy().flatten()
    az_d = np.linspace(-30, 30, 7)
    R = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])

    results = {}
    for i in range(7):
        az = np.radians(az_d[i])
        db = np.array([np.cos(az), np.sin(az), 0])
        dw = R @ db
        hit = rf[i] >= 0
        key = "az=%d (%.0f\u00b0)" % (i, az_d[i])
        results[key] = "HIT r=%.4f" % rf[i] if hit else "MISS"
    gs.destroy()
    return results

# Corridor walls (from echos_mapper.py)
w_outer_bottom = gs.morphs.Box(size=(7.0, 0.01, 2.0), pos=(2.75, -1.5, 1.0), fixed=True)
w_outer_left   = gs.morphs.Box(size=(0.01, 6.0, 2.0), pos=(-0.5, 1.0, 1.0), fixed=True)
w_top          = gs.morphs.Box(size=(3.5, 0.01, 2.0), pos=(1.25, 1.5, 1.0), fixed=True)
w_end          = gs.morphs.Box(size=(3.5, 0.01, 2.0), pos=(4.75, 4.0, 1.0), fixed=True)
w_obstacle     = gs.morphs.Box(size=(0.5, 0.8, 1.5), pos=(1.5, 1.0, 0.75), fixed=True)

all_walls = {
    "none": [],
    "top": [w_top],
    "end": [w_end],
    "outer_bottom": [w_outer_bottom],
    "outer_left": [w_outer_left],
    "obstacle": [w_obstacle],
    "top+end": [w_top, w_end],
    "outer_both": [w_outer_bottom, w_outer_left],
    "all_minus_inner": [w_outer_bottom, w_outer_left, w_top, w_end, w_obstacle],
}

print("=== Isolated raycaster test — adding corridor walls one by one ===")
for name, walls in all_walls.items():
    res = run_test(name, walls)
    print("\n%s:" % name)
    for k, v in res.items():
        print("  %s: %s" % (k, v))
