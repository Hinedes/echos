import numpy as np

cx, cy, cz = 3.0, 2.75, 1.0
dx, dy, dz = 0.005, 1.5, 1.0

verts = [
    (cx-dx, cy-dy, cz-dz),
    (cx+dx, cy-dy, cz-dz),
    (cx+dx, cy+dy, cz-dz),
    (cx-dx, cy+dy, cz-dz),
    (cx-dx, cy-dy, cz+dz),
    (cx+dx, cy-dy, cz+dz),
    (cx+dx, cy+dy, cz+dz),
    (cx-dx, cy+dy, cz+dz),
]

faces_ccw = [
    [0,1,2,3],
    [5,4,7,6],
    [1,5,6,2],
    [4,0,3,7],
    [4,5,1,0],
    [3,2,6,7],
]

def quad_to_tris(q):
    return [[q[0],q[1],q[2]],[q[0],q[2],q[3]]]

def reverse_winding(t):
    return [t[2],t[1],t[0]]

lines = ["# inner-right wall at x=3, y=[1.25,4.25], z=[0,2]"]
lines.append("# double-winding closed prism - every face has both windings")
for v in verts:
    lines.append("v %.6f %.6f %.6f" % v)
for q in faces_ccw:
    tris = quad_to_tris(q)
    for t in tris:
        lines.append("f %d %d %d" % (t[0]+1, t[1]+1, t[2]+1))
    for t in tris:
        r = reverse_winding(t)
        lines.append("f %d %d %d" % (r[0]+1, r[1]+1, r[2]+1))

with open("/workspace/inner_wall.obj","w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote /workspace/inner_wall.obj with %d vertices, %d faces" % (len(verts), len(faces_ccw)*4))
