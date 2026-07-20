import numpy as np

# The offending ray from step 1656, az=6
# emitter=(3.089, 2.828) dir=(-0.151, 0.987) raw_rng=1.000 hit=True max_range=5.0
emitter = np.array([3.089, 2.828, 1.0])
direction = np.array([-0.151, 0.987, 0.0])
raw_range = 1.000
is_hit = True
max_range = 5.0
eps = 0.004

bounds = (-2.0, 7.0, -2.0, 5.0)
res = 0.08

class MapperOld:
    def __init__(self, bounds, resolution=0.08):
        self.res = resolution
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.w = int((self.x_max - self.x_min) / self.res) + 1
        self.h = int((self.y_max - self.y_min) / self.res) + 1
        self.hits = np.zeros((self.h, self.w), dtype=np.int32)
        self.views = np.zeros((self.h, self.w), dtype=np.int32)
        self.clear_r = 5  # old clearance

    def w2g(self, x, y):
        return int((x - self.x_min) / self.res), int((y - self.y_min) / self.res)

    def g2w(self, ix, iy):
        return self.x_min + (ix + 0.5) * self.res, self.y_min + (iy + 0.5) * self.res

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.w and 0 <= iy < self.h

    def update_ray(self, ew, dr, raw_rng, mx, hit, eps=0.004):
        ex, ey = ew[0], ew[1]
        if hit:
            rng = raw_rng + eps
            ex2 = ex + dr[0] * rng
            ey2 = ey + dr[1] * rng
        else:
            rng = mx
            ex2 = ex + dr[0] * rng
            ey2 = ey + dr[1] * rng
        ix0, iy0 = self.w2g(ex, ey)
        ix1, iy1 = self.w2g(ex2, ey2)
        n = max(abs(ix1 - ix0) + 1, abs(iy1 - iy0) + 1)
        cr = self.clear_r
        for i in range(n + 1):
            t = i / max(n, 1)
            cx = int(round(ix0 + t * (ix1 - ix0)))
            cy = int(round(iy0 + t * (iy1 - iy0)))
            term = (i >= n - 1 and hit)
            for ddx in range(-cr, cr + 1):
                for ddy in range(-cr, cr + 1):
                    nx, ny = cx + ddx, cy + ddy
                    if not self.in_bounds(nx, ny):
                        continue
                    if abs(ddx) + abs(ddy) > cr:
                        continue
                    self.views[ny, nx] += 1
                    if term and abs(ddx) + abs(ddy) <= 1:
                        self.hits[ny, nx] += 1


class MapperNew:
    def __init__(self, bounds, resolution=0.08):
        self.res = resolution
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.w = int((self.x_max - self.x_min) / self.res) + 1
        self.h = int((self.y_max - self.y_min) / self.res) + 1
        self.hits = np.zeros((self.h, self.w), dtype=np.int32)
        self.views = np.zeros((self.h, self.w), dtype=np.int32)

    def w2g(self, x, y):
        return int((x - self.x_min) / self.res), int((y - self.y_min) / self.res)

    def g2w(self, ix, iy):
        return self.x_min + (ix + 0.5) * self.res, self.y_min + (iy + 0.5) * self.res

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.w and 0 <= iy < self.h

    def update_ray(self, ew, dr, raw_rng, mx, hit, eps=0.004):
        ex, ey = ew[0], ew[1]
        if hit:
            rng = raw_rng + eps
            ex2 = ex + dr[0] * rng
            ey2 = ey + dr[1] * rng
        else:
            rng = mx
            ex2 = ex + dr[0] * rng
            ey2 = ey + dr[1] * rng
        ix0, iy0 = self.w2g(ex, ey)
        ix1, iy1 = self.w2g(ex2, ey2)
        n = max(abs(ix1 - ix0) + 1, abs(iy1 - iy0) + 1)
        for i in range(n + 1):
            t = i / max(n, 1)
            cx = int(round(ix0 + t * (ix1 - ix0)))
            cy = int(round(iy0 + t * (iy1 - iy0)))
            term = (i >= n - 1 and hit)
            if self.in_bounds(cx, cy):
                self.views[cy, cx] += 1
                if term:
                    self.hits[cy, cx] += 1


# Run both mappers on the offending ray
m_old = MapperOld(bounds, res)
m_new = MapperNew(bounds, res)

m_old.update_ray(emitter, direction, raw_range, max_range, is_hit, eps)
m_new.update_ray(emitter, direction, raw_range, max_range, is_hit, eps)

# Frontier #2 cell
tix, tiy = 56, 81  # (2.52, 4.52)

print(f"Offending ray: emitter=({emitter[0]:.3f},{emitter[1]:.3f}) dir=({direction[0]:.3f},{direction[1]:.3f})")
print(f"raw_range={raw_range:.3f} hit={is_hit}")
print()

# Trace where the ray actually goes
ex, ey = emitter[0], emitter[1]
rng = raw_range + eps
end_x = ex + direction[0] * rng
end_y = ey + direction[1] * rng
ix0, iy0 = m_old.w2g(ex, ey)
ix1, iy1 = m_old.w2g(end_x, end_y)
n = max(abs(ix1 - ix0) + 1, abs(iy1 - iy0) + 1)

print(f"Ray endpoint at ({end_x:.3f},{end_y:.3f})")
print(f"Cells traversed (ix, iy) -> world (x,y):")
ray_cells = []
for i in range(n + 1):
    t = i / max(n, 1)
    cx = int(round(ix0 + t * (ix1 - ix0)))
    cy = int(round(iy0 + t * (iy1 - iy0)))
    wx, wy = m_old.g2w(cx, cy)
    ray_cells.append((cx, cy))
    wall_hit = "(crosses x=3 wall)" if (cx > 62 and i > 0 and ray_cells[-2][0] <= 62) else ""
    print(f"  [{i:2d}] ix={cx:3d} iy={cy:3d} ({wx:.2f},{wy:.2f}){wall_hit}")

print()

# 7x7 neighborhood of frontier #2 cell under OLD mapper
print(f"1. Cells changed by OLD update (cr=5 clearance):")
old_changed = []
for dy in range(-3, 4):
    for dx in range(-3, 4):
        nx, ny = tix + dx, tiy + dy
        if m_old.in_bounds(nx, ny) and m_old.views[ny, nx] > 0:
            wx, wy = m_old.g2w(nx, ny)
            old_changed.append((nx, ny, wx, wy))
            print(f"  ({wx:.2f},{wy:.2f}) views={m_old.views[ny,nx]} hits={m_old.hits[ny,nx]}")

print(f"\nTotal cells touched by OLD in 7x7: {len(old_changed)}")

# 7x7 neighborhood of frontier #2 cell under NEW mapper
print(f"\n2. Cells changed by CORRECTED update (no clearance):")
new_changed = []
for dy in range(-3, 4):
    for dx in range(-3, 4):
        nx, ny = tix + dx, tiy + dy
        if m_new.in_bounds(nx, ny) and m_new.views[ny, nx] > 0:
            wx, wy = m_new.g2w(nx, ny)
            new_changed.append((nx, ny, wx, wy))
            print(f"  ({wx:.2f},{wy:.2f}) views={m_new.views[ny,nx]} hits={m_new.hits[ny,nx]}")

print(f"\nTotal cells touched by NEW in 7x7: {len(new_changed)}")

# Check frontier #2 cell specifically
print(f"\n3. Frontier #2 cell (2.52, 4.52) = ({tix},{tiy}):")
print(f"   OLD: views={m_old.views[tiy,tix]} hits={m_old.hits[tiy,tix]} -> {'FREE' if m_old.views[tiy,tix]>0 and m_old.hits[tiy,tix]==0 else 'UNKNOWN'}")
print(f"   NEW: views={m_new.views[tiy,tix]} hits={m_new.hits[tiy,tix]} -> {'FREE' if m_new.views[tiy,tix]>0 and m_new.hits[tiy,tix]==0 else 'UNKNOWN'}")

# Check wall-crossing
print(f"\n4. Does the ray cross the inner right wall (x=3)?")
t_to_wall = (3.0 - emitter[0]) / direction[0]
if t_to_wall > 0:
    y_at_wall = emitter[1] + direction[1] * t_to_wall
    if 1.5 <= y_at_wall <= 4:
        print(f"   YES: crosses at t={t_to_wall:.3f}, y={y_at_wall:.3f} (within wall y-range [1.5,4])")
        print(f"   Genesis raycaster MISSED the wall (returned range {raw_range:.3f}m vs wall distance {t_to_wall:.3f}m)")
    else:
        print(f"   Crosses at y={y_at_wall:.3f} but wall's y-range is [1.5,4]")
else:
    print(f"   NO: t={t_to_wall:.3f} <= 0")
