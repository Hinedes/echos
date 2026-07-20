import genesis as gs
import numpy as np

results = {"pass": 0, "fail": 0}
report = []

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
    hit = r >= 0
    line = "%s: range=%.4f %s" % (name, r, "HIT" if hit else "MISS")
    ok = True
    if expected is not None:
        if hit:
            err = abs(r - expected)
            ok = err < 0.005
            line += " expected=%.3f err=%.1fmm %s" % (expected, err*1000, "PASS" if ok else "FAIL")
        else:
            ok = False
            line += " expected=%.3f but MISS" % expected
    if ok:
        results["pass"] += 1
    else:
        results["fail"] += 1
    report.append(line)
    gs.destroy()

# Wall at x=3, y=[1.25,4.25], z=[0,2], thickness 0.01m
# +X surface at x=3.005, -X surface at x=2.995
# Expected ranges computed analytically

tests = [
    # Offending ray from the +X side
    ("1. offending -X",    (3.089, 2.828, 1.0), (-0.151, 0.987, 0), 0.556),
    # Perpendicular from +X (outside corridor, toward wall)
    ("2. perp +X side",    (3.500, 2.750, 1.0), (-1.0, 0.0, 0), 0.495),
    # Perpendicular from -X (inside corridor, toward wall)
    ("3. perp -X side",    (2.500, 2.750, 1.0), (1.0, 0.0, 0), 0.495),
    # Oblique from +X (outside, angled approach)
    ("4. oblique +X",      (3.500, 3.500, 1.0), (-0.707, -0.707, 0), 0.700),
    # Oblique from -X (inside, angled approach)
    ("5. oblique -X",      (2.500, 2.500, 1.0), (0.707, 0.707, 0), 0.700),
    # Near top edge from +X
    ("6. near-top +X",     (3.500, 4.200, 1.0), (-1.0, 0.0, 0), 0.495),
    # Near bottom edge from -X
    ("7. near-bot -X",     (2.500, 1.300, 1.0), (1.0, 0.0, 0), 0.495),
    # Miss: above wall y-range
    ("8. miss above y",    (3.500, 5.000, 1.0), (-1.0, 0.0, 0), None),
    # Miss: below wall y-range
    ("9. miss below y",    (3.500, 0.500, 1.0), (-1.0, 0.0, 0), None),
    # Miss: above wall z-range
    ("10. miss above z",   (3.500, 2.750, 2.500), (-1.0, 0.0, 0), None),
    # Far left from -X side (inside corridor, toward +X)
    ("11. far left -X",    (0.500, 2.750, 1.0), (1.0, 0.0, 0), 2.495),
    # Far left from +X side (inside corridor, away from wall)
    ("12. far left +X",    (0.500, 2.750, 1.0), (-1.0, 0.0, 0), None),
    # Tight to bottom edge from -X
    ("13. tight bot -X",   (2.500, 1.260, 1.0), (1.0, 0.0, 0), 0.495),
    # Tight to top edge from -X
    ("14. tight top -X",   (2.500, 4.240, 1.0), (1.0, 0.0, 0), 0.495),
    # Just outside top edge from +X
    ("15. outside top",    (3.500, 4.260, 1.0), (-1.0, 0.0, 0), None),
    # Just outside bottom edge from +X
    ("16. outside bot",    (3.500, 1.240, 1.0), (-1.0, 0.0, 0), None),
    # Raproach from +Y side, aim at wall
    ("17. from +Y side",   (3.000, 5.000, 1.0), (0.0, -1.0, 0), 0.750),
]

for t in tests:
    test_ray(*t)

print("\n==================== RESULTS ====================")
for r in report:
    print(r)
print("\nPass: %d  Fail: %d" % (results["pass"], results["fail"]))
if results["fail"] > 0:
    print("*** FAIL ***")
else:
    print("*** ALL PASS ***")
