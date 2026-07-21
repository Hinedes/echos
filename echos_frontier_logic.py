"""Pure frontier selection and validation helpers."""
from collections import deque

import numpy as np

from echos_core import *


MAP_BOUNDS = (-2.0, 7.0, -2.0, 5.0)
MAP_RESOLUTION = 0.08
FAILED_RETRY_NEW_CELLS = 20

# The deterministic validation world used by the Genesis scene. These bounds
# describe the intended L-shaped flyable region, independent of mapper output.
HORIZONTAL_REGION = (-0.5, 6.25, -1.5, 1.5)
VERTICAL_REGION = (3.0, 6.5, 1.5, 4.0)
OBSTACLE_REGION = (1.25, 1.75, 0.6, 1.4)


def frontier_clusters(occ, inf):
    """Return 8-connected clusters of reachable free/unknown boundary cells."""
    h, w = occ.shape
    frontier = np.zeros((h, w), dtype=bool)
    for iy in range(1, h - 1):
        for ix in range(1, w - 1):
            if occ[iy, ix] != 0.5 or inf[iy, ix] != 0:
                continue
            if (
                occ[iy - 1, ix] == 0.0
                or occ[iy + 1, ix] == 0.0
                or occ[iy, ix - 1] == 0.0
                or occ[iy, ix + 1] == 0.0
            ):
                frontier[iy, ix] = True

    clusters = {}
    visited = np.zeros_like(frontier)
    label = 1
    for iy in range(1, h - 1):
        for ix in range(1, w - 1):
            if not frontier[iy, ix] or visited[iy, ix]:
                continue
            queue = [(ix, iy)]
            visited[iy, ix] = True
            cells = []
            while queue:
                cx, cy = queue.pop()
                cells.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and frontier[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            queue.append((nx, ny))
            clusters[label] = cells
            label += 1
    return clusters


failed_frontiers = {}


def map_revision(mapper):
    """Count cells that have received evidence, not repeated ray traffic."""
    return int(np.count_nonzero(mapper.views))


def clear_failed_frontiers():
    failed_frontiers.clear()


def mark_frontier_failed(ccx, ccy, mapper):
    failed_frontiers[(round(ccx), round(ccy))] = map_revision(mapper)


def _frontier_is_in_cooldown(center, mapper):
    failed_at = failed_frontiers.get(center)
    if failed_at is None:
        return False
    return map_revision(mapper) - failed_at < FAILED_RETRY_NEW_CELLS


def select_frontier(occ, inf, clusters, cur_pos, mapper):
    sx, sy = mapper.w2g(cur_pos[0], cur_pos[1])
    best = None
    best_score = -np.inf
    best_path = None

    for cells in clusters.values():
        if len(cells) < 5:
            continue

        ccx = sum(cell[0] for cell in cells) / len(cells)
        ccy = sum(cell[1] for cell in cells) / len(cells)
        center = (round(ccx), round(ccy))
        if _frontier_is_in_cooldown(center, mapper):
            continue

        adjacent_unknown = set()
        for ix, iy in cells:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = ix + dx, iy + dy
                if (
                    0 <= nx < mapper.w
                    and 0 <= ny < mapper.h
                    and occ[ny, nx] == 0.0
                ):
                    adjacent_unknown.add((nx, ny))
        if not adjacent_unknown:
            continue

        unknown_x = sum(p[0] for p in adjacent_unknown) / len(adjacent_unknown)
        unknown_y = sum(p[1] for p in adjacent_unknown) / len(adjacent_unknown)
        candidates = [
            cell
            for cell in cells
            if occ[cell[1], cell[0]] == 0.5 and inf[cell[1], cell[0]] == 0
        ]
        if not candidates:
            continue
        gx, gy = min(
            candidates,
            key=lambda p: abs(p[0] - unknown_x) + abs(p[1] - unknown_y),
        )

        path = astar_path(inf, (sx, sy), (gx, gy))
        if path is None:
            continue
        path_cost = sum(
            SQRT2
            if path[i][0] != path[i - 1][0] and path[i][1] != path[i - 1][1]
            else 1.0
            for i in range(1, len(path))
        )
        score = len(cells) / (path_cost * mapper.res + 0.01)
        if score <= best_score:
            continue

        cx, cy = mapper.g2w(gx, gy)
        unknown_wx, unknown_wy = mapper.g2w(unknown_x, unknown_y)
        best = {
            "cx": cx,
            "cy": cy,
            "cix": gx,
            "ciy": gy,
            "n": len(cells),
            "ucx": unknown_x,
            "ucy": unknown_y,
            "adj_unk_n": len(adjacent_unknown),
            "ccx": center[0],
            "ccy": center[1],
            "uk_world": (unknown_wx, unknown_wy),
            "observe_heading": np.arctan2(unknown_wy - cy, unknown_wx - cx),
            "score": score,
        }
        best_score = score
        best_path = path

    return best, best_path


def _inside_rect(x, y, rect, margin=0.0):
    x1, x2, y1, y2 = rect
    return x1 + margin <= x <= x2 - margin and y1 + margin <= y <= y2 - margin


def is_ground_truth_free(x, y, clearance=BODY_W / 2 + 0.02):
    """Return whether a body centre is collision-free in the validation world."""
    in_horizontal = _inside_rect(x, y, HORIZONTAL_REGION, clearance)
    vx1, vx2, vy1, vy2 = VERTICAL_REGION
    in_vertical = (
        vx1 + clearance <= x <= vx2 - clearance
        and vy1 - clearance <= y <= vy2 - clearance
    )
    if not (in_horizontal or in_vertical):
        return False

    ox1, ox2, oy1, oy2 = OBSTACLE_REGION
    inside_expanded_obstacle = (
        ox1 - clearance <= x <= ox2 + clearance
        and oy1 - clearance <= y <= oy2 + clearance
    )
    return not inside_expanded_obstacle


def ground_truth_reachable_cells(mapper, launch):
    """Flood-fill true flyable cells without consulting the generated map."""
    start = mapper.w2g(launch[0], launch[1])
    queue = deque([start])
    reachable = set()
    while queue:
        ix, iy = queue.popleft()
        if (ix, iy) in reachable or not mapper.in_b(ix, iy):
            continue
        wx, wy = mapper.g2w(ix, iy)
        if not is_ground_truth_free(wx, wy):
            continue
        reachable.add((ix, iy))
        queue.extend(
            [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)]
        )
    return reachable


def cell_is_known_free(mapper, pos):
    ix, iy = mapper.w2g(pos[0], pos[1])
    return mapper.in_b(ix, iy) and mapper.get_map()[iy, ix] == 0.5


def exploration_metrics(mapper, launch):
    occ = mapper.get_map()
    seen = mapper.views > 0
    reachable = ground_truth_reachable_cells(mapper, launch)
    discovered = {
        (ix, iy)
        for ix, iy in reachable
        if seen[iy, ix] and occ[iy, ix] == 0.5
    }
    coverage = 100.0 * len(discovered) / len(reachable) if reachable else 0.0

    horizontal = {
        cell
        for cell in reachable
        if mapper.g2w(*cell)[1] < 1.5 and mapper.g2w(*cell)[0] < 3.0
    }
    vertical = {
        cell
        for cell in reachable
        if mapper.g2w(*cell)[0] > 3.0 and mapper.g2w(*cell)[1] > 1.5
    }
    horizontal_discovered = len(discovered & horizontal) >= 3
    vertical_discovered = len(discovered & vertical) >= 3
    return (
        coverage,
        horizontal_discovered,
        vertical_discovered,
        len(reachable),
        len(discovered),
    )
