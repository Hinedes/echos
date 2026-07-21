"""CPU-only regression tests for frontier selection and validation."""
import unittest

import numpy as np

from echos_core import OccupancyMapper
from echos_frontier_logic import (
    FAILED_RETRY_NEW_CELLS,
    MAP_BOUNDS,
    MAP_RESOLUTION,
    _frontier_is_in_cooldown,
    clear_failed_frontiers,
    exploration_metrics,
    frontier_clusters,
    ground_truth_reachable_cells,
    map_revision,
    mark_frontier_failed,
)


class FrontierValidationTests(unittest.TestCase):
    def setUp(self):
        self.mapper = OccupancyMapper(MAP_BOUNDS, MAP_RESOLUTION)
        self.launch = np.array([0.0, 0.0, 1.0])

    def test_empty_map_has_zero_coverage_with_nonzero_denominator(self):
        reachable = ground_truth_reachable_cells(self.mapper, self.launch)
        self.assertGreater(len(reachable), 1000)
        coverage, _, _, total, discovered = exploration_metrics(
            self.mapper, self.launch
        )
        self.assertEqual(total, len(reachable))
        self.assertEqual(discovered, 0)
        self.assertEqual(coverage, 0.0)

    def test_fully_observed_ground_truth_map_has_full_coverage(self):
        reachable = ground_truth_reachable_cells(self.mapper, self.launch)
        for ix, iy in reachable:
            self.mapper.views[iy, ix] = 1
            self.mapper.log_odds[iy, ix] = -1
        coverage, horizontal, vertical, total, discovered = exploration_metrics(
            self.mapper, self.launch
        )
        self.assertEqual(total, discovered)
        self.assertEqual(coverage, 100.0)
        self.assertTrue(horizontal)
        self.assertTrue(vertical)

    def test_repeated_rays_do_not_advance_map_revision(self):
        self.mapper.views[4, 4] = 1
        revision = map_revision(self.mapper)
        self.mapper.views[4, 4] += 10000
        self.assertEqual(map_revision(self.mapper), revision)

    def test_failed_frontier_requires_new_cells_before_retry(self):
        clear_failed_frontiers()
        mark_frontier_failed(10, 20, self.mapper)
        self.assertTrue(_frontier_is_in_cooldown((10, 20), self.mapper))
        for i in range(FAILED_RETRY_NEW_CELLS):
            self.mapper.views[0, i] = 1
        self.assertFalse(_frontier_is_in_cooldown((10, 20), self.mapper))

    def test_frontier_cluster_uses_eight_connectivity(self):
        occ = np.zeros((8, 8))
        inflated = np.ones((8, 8))
        cells = [(2, 2), (3, 2), (4, 3), (4, 4), (3, 4)]
        for ix, iy in cells:
            occ[iy, ix] = 0.5
            inflated[iy, ix] = 0.0
        clusters = frontier_clusters(occ, inflated)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(next(iter(clusters.values()))), set(cells))


if __name__ == "__main__":
    unittest.main()
