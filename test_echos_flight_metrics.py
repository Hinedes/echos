"""CPU-only tests for flight validation metrics."""
import unittest

from echos_flight_metrics import disturbance_peak, sustained_recovery_time


class FlightMetricTests(unittest.TestCase):
    def test_never_recovered_returns_none(self):
        errors = [0.2] * 100
        self.assertIsNone(
            sustained_recovery_time(errors, 20, threshold=0.05, hold_steps=10)
        )

    def test_short_threshold_crossing_is_not_recovery(self):
        errors = [0.2] * 20 + [0.01] * 5 + [0.2] * 20
        self.assertIsNone(
            sustained_recovery_time(errors, 20, threshold=0.05, hold_steps=10)
        )

    def test_sustained_recovery_uses_start_of_hold_window(self):
        errors = [0.2] * 25 + [0.01] * 10
        self.assertAlmostEqual(
            sustained_recovery_time(
                errors,
                20,
                threshold=0.05,
                hold_steps=10,
                dt=0.01,
            ),
            0.05,
        )

    def test_already_recovered_is_zero_seconds(self):
        errors = [0.01] * 20
        self.assertEqual(
            sustained_recovery_time(errors, 0, hold_steps=10),
            0.0,
        )

    def test_peak_includes_disturbance_interval(self):
        errors = [0.01, 0.7, 0.2, 0.01]
        self.assertEqual(disturbance_peak(errors, 1), 0.7)

    def test_empty_peak_returns_none(self):
        self.assertIsNone(disturbance_peak([0.1], 1))


if __name__ == "__main__":
    unittest.main()
