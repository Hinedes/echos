"""Reproduce corridor raycaster misses from +X side at close range."""
import genesis as gs
import numpy as np

gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
body = scene.add_entity(gs.morphs.Box(size=(0.165,0.165,0.0545), pos=(0,0,1), fixed=False))
scene.add_entity(gs.morphs.Plane())
# Single wall: the inner-right wall at x=3
scene.add_entity(gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True))
emit_off = (0.0865, 0.0, 0.0)
sensor = scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
    entity_idx=body.idx, pos_offset=(0.0,0.0,0.0),
    euler_offset=(0.0,0.0,0.0), max_range=5.0, no_hit_value=-1.0))
scene.build()
body.set_mass(0.225)

# Reproduce corridor step 1652 az=6 conditions
# Body pos such that sensor origin (with offset) is at (3.077, 2.794, 1.0)
# Body yaw = 75° -> forward = (cos75, sin75) = (0.259, 0.966)
# Sensor origin = body_pos + 0.0865 * forward
# We want sensor at (3.077, 2.794, 1.0)
# body_pos = (3.077, 2.794, 1.0) - 0.0865 * (0.259, 0.966) = (3.055, 2.710, 1.0)
yaw = np.radians(75)
fwd = np.array([np.cos(yaw), np.sin(yaw), 0])
body_pos = np.array([3.077, 2.794, 1.0]) - 0.0865 * fwd
body.set_pos(body_pos)
body.set_quat(np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))

# Check: sensor origin with forward offset should be at (3.077, 2.794, 1.0)
R = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
sensor_origin = body_pos + R @ np.array([0.0865, 0, 0])
print("Sensor origin: (%.3f, %.3f, %.3f)" % tuple(sensor_origin))
print("Expected:      (3.077, 2.794, 1.000)")
print("Body pos:      (%.3f, %.3f, %.3f)" % tuple(body_pos))

# Test the offending beam at +30° from body-forward
# In body frame: (cos30, sin30, 0) = (0.866, 0.5, 0)
# In world: R @ (0.866, 0.5)
beam_body = np.array([np.cos(np.radians(30)), np.sin(np.radians(30)), 0])
beam_world = R @ beam_body
print("Beam world dir: (%.3f, %.3f, %.3f)" % tuple(beam_world))

# Now we need to test with a forward offset
gs.destroy()
del scene

# Create a simplified test with pos_offset
gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
body2 = scene.add_entity(gs.morphs.Box(size=(0.165,0.165,0.0545), pos=(0,0,1), fixed=False))
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True))
sensor2 = scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(angles=(np.array([0.0]), np.array([0.0]))),
    entity_idx=body2.idx, pos_offset=(0.0865, 0.0, 0.0),
    euler_offset=(0.0, 0.0, 0.0), max_range=5.0, no_hit_value=-1.0))
scene.build()
body2.set_mass(0.225)

# Position body2 so sensor origin is at the exact offending ray origin
# with the correct yaw. The sensor offset (0.0865,0,0) is in body frame.
# Sensor world = body_pos + R @ (0.0865, 0, 0)
# We want sensor at (3.077, 2.794, 1.0) with body yaw=75°
# body_pos = (3.077, 2.794, 1.0) - R @ (0.0865, 0, 0)
body2.set_pos(body_pos)
body2.set_quat(np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))

# Now read sensor (forward direction = body forward, but beam at +30° isn't available
# since we have a single-beam sensor. Let me use the actual 7-beam sensor.

gs.destroy()
del scene

# Final test: full corridor with ARGUS at the exact step 1652 configuration
gs.init(backend=gs.amdgpu)
scene = gs.Scene(show_viewer=False, rigid_options=gs.options.RigidOptions(enable_collision=True))
body3 = scene.add_entity(gs.morphs.Box(size=(0.165,0.165,0.0545), pos=(0,0,1), fixed=False))
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True))
# Also add the offending ray test WITHOUT the corridor clutter
sensor3 = scene.add_sensor(gs.sensors.Raycaster(
    pattern=gs.sensors.SphericalPattern(fov=(60.0,0.0), n_points=(7,1)),
    entity_idx=body3.idx, pos_offset=(0.0865, 0.0, 0.0),
    euler_offset=(0.0, 0.0, 0.0), max_range=5.0, no_hit_value=-1.0))
scene.build()
body3.set_mass(0.225)

# Position and yaw same as before
body3.set_pos(body_pos)
body3.set_quat(np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))
for _ in range(10): scene.step()

d = sensor3.read()
rf = d.distances.cpu().numpy().flatten()
az_d = np.linspace(-30, 30, 7)
R = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
ew = body3.get_pos().cpu().numpy() + R @ np.array([0.0865, 0, 0])

print("\n=== Step 1652 repro: wall+plane only (no corridor clutter) ===")
for i in range(7):
    az = np.radians(az_d[i])
    db = np.array([np.cos(az), np.sin(az), 0])
    dw = R @ db
    hit = rf[i] >= 0
    print(f"  az={i} ({az_d[i]:.0f}°): {'HIT' if hit else 'MISS'} range={rf[i]:.4f} dir=({dw[0]:.3f},{dw[1]:.3f})")
    if not hit and dw[0] < 0:
        # Print expected wall distance
        wall_t = (3.005 - ew[0]) / dw[0]
        print(f"    emitter x={ew[0]:.3f} wall_x={3.005:.3f} wall_t={wall_t:.3f} (in range: {0 < wall_t <= 5})")

gs.destroy()
