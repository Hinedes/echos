"""Genesis scene and sensor helpers for frontier exploration."""
import numpy as np

from echos_core import *


def _build_scene():
    import genesis as gs

    gs.init(backend=gs.amdgpu)
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(enable_collision=True),
    )
    body = scene.add_entity(
        gs.morphs.Box(size=(BODY_W, BODY_D, BODY_H), pos=(0, 0, 1), fixed=False)
    )
    scene.add_entity(
        gs.morphs.Box(size=(7.0, 0.01, 2.0), pos=(2.75, -1.5, 1.0), fixed=True)
    )
    scene.add_entity(
        gs.morphs.Box(size=(0.01, 6.0, 2.0), pos=(-0.5, 1.0, 1.0), fixed=True)
    )
    scene.add_entity(
        gs.morphs.Box(size=(3.5, 0.01, 2.0), pos=(1.25, 1.5, 1.0), fixed=True)
    )
    scene.add_entity(
        gs.morphs.Box(size=(0.01, 3.0, 2.0), pos=(3.0, 2.75, 1.0), fixed=True)
    )
    scene.add_entity(
        gs.morphs.Box(size=(3.5, 0.01, 2.0), pos=(4.75, 4.0, 1.0), fixed=True)
    )
    scene.add_entity(
        gs.morphs.Box(size=(0.5, 0.8, 1.5), pos=(1.5, 1.0, 0.75), fixed=True)
    )
    scene.add_entity(gs.morphs.Plane())

    flood = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.SphericalPattern(fov=(60.0, 0.0), n_points=(7, 1)),
            entity_idx=body.idx,
            pos_offset=(RAYCAST_ORIGIN, 0.0, 0.0),
            euler_offset=(0.0, 0.0, 0.0),
            max_range=5.0,
            no_hit_value=-1.0,
        )
    )
    throw = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.SphericalPattern(
                angles=(np.array([0.0]), np.array([0.0]))
            ),
            entity_idx=body.idx,
            pos_offset=(RAYCAST_ORIGIN, 0.0, 0.0),
            euler_offset=(0.0, 0.0, 0.0),
            max_range=8.0,
            no_hit_value=-1.0,
        )
    )
    scene.build()
    body.set_mass(MASS)
    return gs, scene, body, flood, throw, scene.sim.rigid_solver, 0


def _scan_into_map(mapper, flood, throw, q_cur, pos, az_deg, step):
    flood_data = flood.read()
    raw_flood = flood_data.distances.cpu().numpy().flatten()
    raw_throw = None
    if step % 5 == 0:
        raw_throw = throw.read().distances.flatten()[0].item()

    Rwb = R_world_from_body(q_cur)
    emitter_world = pos + Rwb @ np.array([RAYCAST_ORIGIN, 0.0, 0.0])
    for i, azimuth in enumerate(az_deg):
        az = np.radians(azimuth)
        direction_world = Rwb @ np.array([np.cos(az), np.sin(az), 0.0])
        hit = raw_flood[i] >= 0.0
        mapper.update_ray(
            emitter_world,
            direction_world,
            raw_flood[i] if hit else 0.0,
            5.0,
            hit,
        )
    if raw_throw is not None:
        direction_world = Rwb @ np.array([1.0, 0.0, 0.0])
        hit = raw_throw >= 0.0
        mapper.update_ray(
            emitter_world,
            direction_world,
            raw_throw if hit else 0.0,
            8.0,
            hit,
        )
    return raw_flood
