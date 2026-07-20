import genesis as gs, numpy as np

results = []

def test_ray(name, emitter, direction, expected=None):
    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
    body = scene.add_entity(gs.morphs.Box(size=(0.01,0.01,0.01), pos=(0,0,1), fixed=True))
    floor = scene.add_entity(gs.morphs.Plane())
    wall = scene.add_entity(gs.morphs.Mesh(file="/workspace/inner_wall.obj", fixed=True))
    sensor = scene.add_sensor(gs.sensors.Raycaster(
        pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
        entity_idx=body.idx, pos_offset=(emitter[0], emitter[1], emitter[2]),
        euler_offset=(0,0,0), max_range=5.0, no_hit_value=-1.0))
    scene.build()
    body.set_mass(0.225)
    theta = np.arctan2(direction[1], direction[0])
    q = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
    body.set_quat(q)
    for _ in range(5): scene.step()
    d = sensor.read()
    r = d.distances.flatten()[0].item()
    gs.destroy()
    
    hit = r >= 0
    line = "%s: range=%.4f %s" % (name, r, "HIT" if hit else "MISS")
    if expected is not None:
        err = abs(r - expected) if hit else 999
        ok = err < 0.005
        line += " expected=%.3f err=%.1fmm %s" % (expected, err*1000, "PASS" if ok else "FAIL")
    print(line)

d=0.25  # wall thickness
test_ray("offending -X",     (3.089, 2.828, 1.0), (-0.151, 0.987, 0), 0.589-0.125+0.125)
# For 0.25m thick wall centered at x=3: surface at x=2.875 (front) and x=3.125 (back)
test_ray("perp +X side",    (3.500, 2.750, 1.0), (-1.0, 0.0, 0), 0.375)
test_ray("perp -X side",    (2.500, 2.750, 1.0), (1.0, 0.0, 0), 0.375)
test_ray("oblique +X+Y",    (3.500, 3.500, 1.0), (-0.707, -0.707, 0), 0.530)
test_ray("oblique -X-Y",    (2.500, 2.500, 1.0), (0.707, 0.707, 0), 0.530)
test_ray("miss above y=5",  (3.500, 5.000, 1.0), (-1.0, 0.0, 0))
test_ray("miss below y=0.5",(3.500, 0.500, 1.0), (-1.0, 0.0, 0))
test_ray("far left hit",    (0.500, 2.750, 1.0), (1.0, 0.0, 0), 2.375)
