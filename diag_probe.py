import genesis as gs
import numpy as np

with open("/workspace/thick_wall.obj", "w") as f:
    f.write("# thick wall at x=3, y=[1.25,4.25], z=[0,2], 0.10m thick\n")
    f.write("v 2.950000 1.250000 0.000000\n")
    f.write("v 3.050000 1.250000 0.000000\n")
    f.write("v 3.050000 4.250000 0.000000\n")
    f.write("v 2.950000 4.250000 0.000000\n")
    f.write("v 2.950000 1.250000 2.000000\n")
    f.write("v 3.050000 1.250000 2.000000\n")
    f.write("v 3.050000 4.250000 2.000000\n")
    f.write("v 2.950000 4.250000 2.000000\n")
    f.write("f 1 2 6\nf 1 6 5\n")
    f.write("f 4 8 7\nf 4 7 3\n")
    f.write("f 2 6 5\nf 2 5 1\n")
    f.write("f 1 5 8\nf 1 8 4\n")
    f.write("f 5 6 7\nf 5 7 4\n")
    f.write("f 1 4 3\nf 1 3 2\n")

with open("/workspace/thin_wall.obj", "w") as f:
    f.write("# thin wall at x=3, y=[1.25,4.25], z=[0,2], 0.01m thick\n")
    f.write("v 2.995000 1.250000 0.000000\n")
    f.write("v 3.005000 1.250000 0.000000\n")
    f.write("v 3.005000 4.250000 0.000000\n")
    f.write("v 2.995000 4.250000 0.000000\n")
    f.write("v 2.995000 1.250000 2.000000\n")
    f.write("v 3.005000 1.250000 2.000000\n")
    f.write("v 3.005000 4.250000 2.000000\n")
    f.write("v 2.995000 4.250000 2.000000\n")
    f.write("f 1 2 6\nf 1 6 5\n")
    f.write("f 4 8 7\nf 4 7 3\n")
    f.write("f 2 6 5\nf 2 5 1\n")
    f.write("f 1 5 8\nf 1 8 4\n")
    f.write("f 5 6 7\nf 5 7 4\n")
    f.write("f 1 4 3\nf 1 3 2\n")

# Use a tiny body with zero offset to minimize self-hit
TINY = 0.0001
SELF_HIT = TINY / 2

def run_test(label, wall_type, wall_arg, tests):
    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
    body = scene.add_entity(gs.morphs.Box(size=(TINY,TINY,TINY), pos=(0,0,0.001), fixed=False))
    scene.add_entity(gs.morphs.Plane())
    if wall_type == "box_prim":
        scene.add_entity(gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True))
    elif wall_type == "mesh":
        scene.add_entity(gs.morphs.Mesh(file=wall_arg, fixed=True))
    sensor = scene.add_sensor(gs.sensors.Raycaster(
        pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
        entity_idx=body.idx, pos_offset=(0.0,0.0,0.0),
        euler_offset=(0.0,0.0,0.0), max_range=5.0, no_hit_value=-1.0))
    scene.build()
    body.set_mass(0.225)

    print("%s:" % label)
    for tname, emitter, direction, expected in tests:
        body.set_pos(emitter)
        theta = np.arctan2(direction[1], direction[0])
        body.set_quat(np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)]))
        for _ in range(5): scene.step()
        d = sensor.read()
        r = d.distances.flatten()[0].item()
        hit = r >= 0 and r > SELF_HIT * 2  # ignore self-hit
        if not hit:
            r_disp = -1.0
        else:
            r_disp = r
        if expected is None:
            ok = not hit
            status = "PASS(miss)" if ok else "FAIL(hit r=%.4f)" % r
        elif hit:
            err = abs(r - expected)
            ok = err < 0.01
            status = "PASS" if ok else "FAIL(err=%.1fmm)" % (err*1000)
        else:
            ok = False
            status = "FAIL(MISS exp=%.3f)" % expected
        print("  %s: r=%s %s" % (tname,
            ("%.4f" % r_disp) if r_disp >= 0 else "MISS", status))
    gs.destroy()

run_test("A. Box primitive 0.01m @ x=3", "box_prim", None, [
    ("perp +X side", (3.5, 2.75, 1.0), (-1.0, 0.0, 0.0), 0.495),
    ("perp -X side", (2.5, 2.75, 1.0), (1.0, 0.0, 0.0), 0.495),
    ("offending -X", (3.089, 2.828, 1.0), (-0.151, 0.987, 0.0), 0.556),
])

run_test("B. Thick mesh 0.10m @ x=3", "mesh", "/workspace/thick_wall.obj", [
    ("perp +X side", (3.5, 2.75, 1.0), (-1.0, 0.0, 0.0), 0.450),
    ("perp -X side", (2.5, 2.75, 1.0), (1.0, 0.0, 0.0), 0.450),
    ("offending +X", (3.089, 2.828, 1.0), (-0.151, 0.987, 0.0), 0.258),
    ("miss above y", (3.5, 5.000, 1.0), (-1.0, 0.0, 0.0), None),
    ("miss below y", (3.5, 0.500, 1.0), (-1.0, 0.0, 0.0), None),
])

run_test("C. Thin single-winding mesh 0.01m @ x=3", "mesh", "/workspace/thin_wall.obj", [
    ("perp +X side", (3.5, 2.75, 1.0), (-1.0, 0.0, 0.0), 0.495),
    ("perp -X side", (2.5, 2.75, 1.0), (1.0, 0.0, 0.0), 0.495),
    ("offending +X", (3.089, 2.828, 1.0), (-0.151, 0.987, 0.0), 0.556),
    ("miss above y", (3.5, 5.000, 1.0), (-1.0, 0.0, 0.0), None),
])
