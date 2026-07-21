"""CPU-only regression tests for echos_core.

Run with:
    python -m unittest -v test_echos_core.py
"""
import math
import unittest

import numpy as np

from echos_core import (
    OccupancyMapper,
    R_world_from_body,
    astar_path,
    mixer,
    quat_error_angle,
    quat_from_z_yaw,
    quat_to_euler,
    wrench_from_thrusts,
)


class QuaternionTests(unittest.TestCase):
    def test_double_cover_has_zero_error(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(quat_error_angle(identity), 0.0, places=12)
        self.assertAlmostEqual(quat_error_angle(-identity), 0.0, places=12)

    def test_rotation_normalizes_input(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R_world_from_body(q), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(quat_to_euler(q), np.zeros(3), atol=1e-12)

    def test_singular_yaw_basis_is_finite_and_orthonormal(self):
        q = quat_from_z_yaw(np.array([0.0, 1.0, 0.0]), math.pi / 2.0)
        self.assertTrue(np.all(np.isfinite(q)))
        R = R_world_from_body(q)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(R[:, 2], np.array([0.0, 1.0, 0.0]), atol=1e-12)


class SupercoverTests(unittest.TestCase):
    def setUp(self):
        self.mapper = OccupancyMapper((-10.0, 10.0, -10.0, 10.0), resolution=1.0)

    def test_diagonal_includes_corner_adjacent_cells(self):
        cells = self.mapper._supercover(0, 0, 2, 2)
        self.assertEqual(
            cells,
            [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)],
        )

    def test_all_octants_are_unique_and_reversible(self):
        targets = [(5, 2), (2, 5), (-2, 5), (-5, 2), (-5, -2), (-2, -5), (2, -5), (5, -2)]
        for target in targets:
            with self.subTest(target=target):
                forward = self.mapper._supercover(0, 0, *target)
                reverse = self.mapper._supercover(*target, 0, 0)
                self.assertEqual(len(forward), len(set(forward)))
                self.assertEqual(set(forward), set(reverse))
                self.assertEqual(forward[0], (0, 0))
                self.assertEqual(forward[-1], target)

    def test_ray_marks_all_corner_cells_viewed(self):
        mapper = OccupancyMapper((0.0, 3.0, 0.0, 3.0), resolution=1.0)
        mapper.update_ray(
            emitter=np.array([0.5, 0.5, 0.0]),
            direction=np.array([1.0, 1.0, 0.0]),
            raw_range=2.0,
            max_range=2.0,
            is_hit=False,
        )
        for ix, iy in mapper._supercover(0, 0, 2, 2):
            self.assertGreater(mapper.views[iy, ix], 0)


class MixerAndPathTests(unittest.TestCase):
    def test_mixer_round_trip_without_saturation(self):
        requested = np.array([0.8, 0.004, -0.003, 0.002])
        thrusts, saturated, _ = mixer(*requested)
        self.assertFalse(np.any(saturated))
        T, tau = wrench_from_thrusts(thrusts)
        np.testing.assert_allclose(np.r_[T, tau], requested, atol=1e-12)

    def test_astar_does_not_cut_blocked_corner(self):
        grid = np.zeros((3, 3))
        grid[0, 1] = 1.0
        grid[1, 0] = 1.0
        self.assertIsNone(astar_path(grid, (0, 0), (1, 1)))

    def test_astar_returns_valid_open_path(self):
        grid = np.zeros((4, 4))
        grid[1, 1] = 1.0
        path = astar_path(grid, (0, 0), (3, 3))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 3))
        self.assertNotIn((1, 1), path)


if __name__ == "__main__":
    unittest.main()
