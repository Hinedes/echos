import genesis as gs

gs.init(backend=gs.amdgpu)

scene = gs.Scene(show_viewer=False)

scene.add_entity(gs.morphs.Plane())

box = scene.add_entity(
    gs.morphs.Box(
        size=(0.2, 0.2, 0.2),
        pos=(0.0, 0.0, 1.0),
    )
)

scene.build()

print("start:", box.get_pos())

for _ in range(200):
    scene.step()

print("end:", box.get_pos())
print("GENESIS_AMDGPU_SMOKE_PASS")
