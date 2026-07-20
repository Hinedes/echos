"""Shared geometry, dynamics, controller, mapper, and path utilities."""
import math
import numpy as np
import heapq

# ---------------------------------------------------------------------------
# Physical constants & geometry
# ---------------------------------------------------------------------------
BODY_W = 0.165
BODY_D = 0.165
BODY_H = 0.0545
MASS = 0.225
G = 9.81
ARM = 0.0511
KP_ATT = 0.06
KD_ATT = 0.018
KP_POS = 3.5
KD_POS = 3.6
MAX_TILT_DEG = 20.0
MAX_HACC = G * np.tan(np.radians(MAX_TILT_DEG))
MAX_THRUST = 2.0 * MASS * G        # per-rotor saturation (~4.4 N)
BOOTSTRAP_DIST = 2.0                # metres forward for initial fly
MAX_YAW_RATE = 3.0                  # rad/s maximum commanded yaw rate

ROTOR_GEOM = [
    np.array([ARM, -ARM, 0.0]),
    np.array([ARM,  ARM, 0.0]),
    np.array([-ARM, -ARM, 0.0]),
    np.array([-ARM,  ARM, 0.0]),
]
YAW_SIGNS = [1, -1, -1, 1]

# Emitter offsets
PHYSICAL_EMITTER = BODY_W / 2                           # 0.0825 m
RAYCAST_ORIGIN = BODY_W / 2 + 0.004                     # 0.0865 m

# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------
def R_world_from_body(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])

def R_body_from_world(q):
    return R_world_from_body(q).T

# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------
def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    cr = math.cos(math.radians(roll_deg/2))
    sr = math.sin(math.radians(roll_deg/2))
    cp = math.cos(math.radians(pitch_deg/2))
    sp = math.sin(math.radians(pitch_deg/2))
    cy = math.cos(math.radians(yaw_deg/2))
    sy = math.sin(math.radians(yaw_deg/2))
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])

def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2,1] - R[1,2]) * s, (R[0,2] - R[2,0]) * s, (R[1,0] - R[0,1]) * s])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + R[0,0] - R[1,1] - R[2,2]))
        return np.array([(R[2,1] - R[1,2]) / s, 0.25 * s, (R[0,1] + R[1,0]) / s, (R[0,2] + R[2,0]) / s])
    if R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + R[1,1] - R[0,0] - R[2,2]))
        return np.array([(R[0,2] - R[2,0]) / s, (R[0,1] + R[1,0]) / s, 0.25 * s, (R[1,2] + R[2,1]) / s])
    s = 2.0 * np.sqrt(max(0.0, 1.0 + R[2,2] - R[0,0] - R[1,1]))
    return np.array([(R[1,0] - R[0,1]) / s, (R[0,2] + R[2,0]) / s, (R[1,2] + R[2,1]) / s, 0.25 * s])

def quat_from_z_yaw(z_des, psi_des):
    x_c = np.array([np.cos(psi_des), np.sin(psi_des), 0.0])
    zn = z_des / np.linalg.norm(z_des)
    y_des = np.cross(zn, x_c)
    yn = np.linalg.norm(y_des)
    if yn < 1e-10:
        y_des = np.array([0.0, 1.0, 0.0])
    else:
        y_des = y_des / yn
    x_des = np.cross(y_des, zn)
    return quat_from_R(np.column_stack([x_des, y_des, zn]))

def quat_error_angle(qe):
    return 2.0 * np.arccos(np.clip(qe[0], -1.0, 1.0))

def quat_attitude_error(qd, qc):
    return quat_mul(quat_conj(qd), qc)

# ---------------------------------------------------------------------------
# Yaw angle utilities
# ---------------------------------------------------------------------------
def angle_diff(target, current):
    """Signed shortest angular difference in radians."""
    return math.atan2(math.sin(target - current), math.cos(target - current))

def move_toward_angle(current, target, max_step):
    """Move current toward target by at most max_step (radians)."""
    diff = angle_diff(target, current)
    step = np.clip(diff, -max_step, max_step)
    return current + step

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
def position_pd(pos_des, pos_cur, vel_cur, psi_des=0.0):
    pos_err = pos_des - pos_cur
    a_des = KP_POS * pos_err - KD_POS * vel_cur
    a_des[0] = np.clip(a_des[0], -MAX_HACC, MAX_HACC)
    a_des[1] = np.clip(a_des[1], -MAX_HACC, MAX_HACC)
    T_vec = MASS * (a_des + np.array([0.0, 0.0, G]))
    T_total = np.linalg.norm(T_vec)
    if T_total < 1e-6:
        return MASS * G, np.array([1.0, 0.0, 0.0, 0.0])
    return T_total, quat_from_z_yaw(T_vec / T_total, psi_des)

def attitude_pd(q_des, q_cur, omega_world):
    """Returns torque_triad, quat_error."""
    qe = quat_attitude_error(q_des, q_cur)
    sign = 1.0 if qe[0] >= 0.0 else -1.0
    e_R = 2.0 * sign * qe[1:4]
    omega_body = R_body_from_world(q_cur) @ omega_world
    tau = -KP_ATT * e_R - KD_ATT * omega_body
    return tau, qe

def mixer(T, tx, ty, tz=0.0):
    """Returns (thrusts, saturated_mask, thrusts_raw).
    Clips negative thrusts to zero. Upper limit is MAX_THRUST."""
    inv = 1.0 / (4.0 * ARM)
    ts_raw = np.array([
        T/4.0 - tx*inv - ty*inv + tz*inv,
        T/4.0 + tx*inv - ty*inv - tz*inv,
        T/4.0 - tx*inv + ty*inv - tz*inv,
        T/4.0 + tx*inv + ty*inv + tz*inv,
    ])
    ts = np.clip(ts_raw, 0.0, MAX_THRUST)
    saturated = ts != ts_raw
    return ts, saturated, ts_raw

def mixer_with_authority(T, tx, ty, tz=0.0):
    """Mixer with yaw-torque limiting to preserve roll/pitch authority.
    Returns (thrusts, saturated_mask, achieved_torque, thrusts_raw)."""
    ts, sat, raw = mixer(T, tx, ty, tz)
    if not np.any(sat):
        return ts, sat, np.array([tx, ty, tz]), raw
    # Yaw torque has lowest priority: recompute with tz reduced
    tz_reduced = tz * 0.5
    ts2, sat2, raw2 = mixer(T, tx, ty, tz_reduced)
    if not np.any(sat2):
        return ts2, sat2, np.array([tx, ty, tz_reduced]), raw2
    # If still saturated, eliminate yaw torque entirely
    ts3, sat3, raw3 = mixer(T, tx, ty, 0.0)
    return ts3, sat3, np.array([tx, ty, 0.0]), raw3

def apply_rotor_forces(rigid_solver, link_idx, thrusts):
    F_total = np.zeros(3)
    tau_total = np.zeros(3)
    for i, (r, f) in enumerate(zip(ROTOR_GEOM, thrusts)):
        F = np.array([0.0, 0.0, f])
        tau_total += np.cross(r, F)
        tau_total[2] += ARM * YAW_SIGNS[i] * f
        F_total += F
    rigid_solver.apply_links_external_force(
        force=F_total.reshape(1, 3), links_idx=[link_idx], ref="link_origin", local=True,
    )
    rigid_solver.apply_links_external_torque(
        torque=tau_total.reshape(1, 3), links_idx=[link_idx], ref="link_origin", local=True,
    )

# ---------------------------------------------------------------------------
# Occupancy Grid Mapper
# ---------------------------------------------------------------------------
OCC_WEIGHT = 5      # weight of a hit observation
FREE_WEIGHT = 1     # weight of a free observation

class OccupancyMapper:
    def __init__(self, bounds, resolution=0.08):
        self.res = resolution
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.w = int((self.x_max - self.x_min) / self.res) + 1
        self.h = int((self.y_max - self.y_min) / self.res) + 1
        self.hits = np.zeros((self.h, self.w), dtype=np.int32)
        self.views = np.zeros((self.h, self.w), dtype=np.int32)
        self.log_odds = np.zeros((self.h, self.w), dtype=np.float32)

    def w2g(self, x, y):
        """World to grid, using math.floor for correct negative handling."""
        return math.floor((x - self.x_min) / self.res), math.floor((y - self.y_min) / self.res)

    def g2w(self, ix, iy):
        return self.x_min + (ix + 0.5) * self.res, self.y_min + (iy + 0.5) * self.res

    def in_b(self, ix, iy):
        return 0 <= ix < self.w and 0 <= iy < self.h

    def update_ray(self, emitter, direction, raw_range, max_range, is_hit):
        """Single-ray update with supercover traversal.
        eps is NOT added — raw_range is the Genesis-measured distance.
        All traversed cells except the terminal one are free observations.
        The terminal cell (when is_hit) is an occupied observation.
        """
        ex, ey = emitter[0], emitter[1]
        if is_hit:
            rng = raw_range
        else:
            rng = max_range
        end_x = ex + direction[0] * rng
        end_y = ey + direction[1] * rng

        ix0, iy0 = self.w2g(ex, ey)
        ix1, iy1 = self.w2g(end_x, end_y)

        # Supercover line traversal (Bresenham-class)
        dx = abs(ix1 - ix0)
        dy = abs(iy1 - iy0)
        n = max(dx, dy)
        if n == 0:
            return

        step_x = 1 if ix1 >= ix0 else -1
        step_y = 1 if iy1 >= iy0 else -1
        cx, cy = ix0, iy0

        for i in range(n + 1):
            t = i / max(n, 1)
            # integer position along the line
            if n > 0:
                cx = int(round(ix0 + t * (ix1 - ix0)))
                cy = int(round(iy0 + t * (iy1 - iy0)))
            if not self.in_b(cx, cy):
                continue
            if i == n:
                if is_hit:
                    self.log_odds[cy, cx] += OCC_WEIGHT
                    self.hits[cy, cx] += 1
                self.views[cy, cx] += 1
            else:
                self.log_odds[cy, cx] -= FREE_WEIGHT
                self.views[cy, cx] += 1

    def get_map(self):
        occ = np.zeros((self.h, self.w))
        # Occupied: log_odds > 0 (hits dominate) AND at least one hit
        occ[(self.log_odds > 0) & (self.hits >= 1)] = 1.0
        # Free: log_odds < 0 (views dominate) AND no hits
        occ[(self.log_odds < 0) & (self.hits == 0)] = 0.5
        # If log_odds == 0 or mixed evidence, keep as unknown (0.0)
        return occ

    def get_views(self):
        return self.views

    def get_hits(self):
        return self.hits

# ---------------------------------------------------------------------------
# A* path planner (octile heuristic)
# ---------------------------------------------------------------------------
SQRT2 = math.sqrt(2.0)

def astar_path(inflation_grid, start, goal):
    h, w = inflation_grid.shape
    sx, sy = start
    gx, gy = goal

    def free(ix, iy):
        return 0 <= ix < w and 0 <= iy < h and inflation_grid[iy, ix] == 0

    if not free(sx, sy) or not free(gx, gy):
        return None

    def heuristic(ix, iy):
        dx = abs(ix - gx)
        dy = abs(iy - gy)
        return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)

    opens = [(heuristic(sx, sy), (sx, sy))]
    came = {}
    g_c = {(sx, sy): 0}

    while opens:
        _, cur = heapq.heappop(opens)
        if cur == (gx, gy):
            pth = []
            c = cur
            while c in came:
                pth.append(c)
                c = came[c]
            pth.append((sx, sy))
            pth.reverse()
            return pth

        cx, cy = cur
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = cx + dx, cy + dy
            if not free(nx, ny):
                continue
            if dx != 0 and dy != 0:
                if not free(cx + dx, cy) or not free(cx, cy + dy):
                    continue
            cost = SQRT2 if (dx != 0 and dy != 0) else 1.0
            ng = g_c[cur] + cost
            if (nx, ny) not in g_c or ng < g_c[(nx, ny)]:
                came[(nx, ny)] = cur
                g_c[(nx, ny)] = ng
                heapq.heappush(opens, (ng + heuristic(nx, ny), (nx, ny)))
    return None

# ---------------------------------------------------------------------------
# Obstacle inflation
# ---------------------------------------------------------------------------
def inflation_grid(mapper, inflate_r=4):
    """Build inflation grid from mapper occupancy.
    Known-free cells start traversable. Occupied cells dilated into free only.
    Unknown cells blocked. Trajectory history is NOT force-cleared."""
    occ = mapper.get_map()
    h, w = occ.shape
    inf = np.ones((h, w))
    for iy in range(h):
        for ix in range(w):
            if occ[iy, ix] == 0.5:
                inf[iy, ix] = 0.0
    for iy in range(h):
        for ix in range(w):
            if occ[iy, ix] == 1.0:
                for dy in range(-inflate_r, inflate_r + 1):
                    for dx in range(-inflate_r, inflate_r + 1):
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < w and 0 <= ny < h and occ[ny, nx] == 0.5:
                            inf[ny, nx] = 1.0
    inf[occ == 0.0] = 1.0
    return inf

# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------
def check_collision(body, body_h=BODY_H):
    """Check ground collision using body altitude and half-height.
    Returns True if the body is at or below ground level.
    (Genesis contact query not yet available — altitude sentinel only.)"""
    pos = body.get_pos().cpu().numpy()
    z_thresh = body_h / 2 + 0.005
    return pos[2] <= z_thresh

# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------
def reset_body_state(body, rigid_solver, pos=(0.0, 0.0, 1.0), quat=None):
    """Reset body position, orientation, and external forces.
    Genesis does not expose set_vel/set_ang — momentum carries over."""
    body.set_pos(pos)
    body.set_quat(quat if quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
    rigid_solver.clear_external_force()

# ---------------------------------------------------------------------------
# Bootstrap target
# ---------------------------------------------------------------------------
def bootstrap_target(launch, dist=BOOTSTRAP_DIST):
    """Fixed bootstrap target: dist metres forward from launch."""
    return np.array([launch[0] + dist, launch[1], launch[2]])
