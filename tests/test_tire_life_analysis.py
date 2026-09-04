import unittest

import numpy as np

from tire_life_analysis import (
    analyze_tire_life,
    detect_hybrid_performance_cliff,
    detect_performance_cliff,
    detect_piecewise_performance_cliff,
    detect_rolling_sustained_performance_cliff,
    estimate_useful_life_uncertainty,
    recommend_strategy_useful_life,
)


class TireCliffDetectorTest(unittest.TestCase):
    def test_flat_curve_has_no_cliff(self):
        result = detect_piecewise_performance_cliff(np.full(30, 90.0))
        self.assertIsNone(result["performance_cliff_lap"])

    def test_short_curve_is_rejected(self):
        result = detect_hybrid_performance_cliff(np.linspace(90.0, 90.2, 8))
        self.assertIsNone(result["performance_cliff_lap"])

    def test_noisy_known_breakpoint_is_detectable(self):
        rng = np.random.default_rng(42)
        ages = np.arange(30)
        curve = np.where(ages < 12, 90.0 + .02 * ages, 90.24 + .20 * (ages - 12))
        result = detect_piecewise_performance_cliff(curve + rng.normal(0, .015, len(ages)))
        self.assertIsNotNone(result["performance_cliff_lap"])
    def test_detectors_return_no_cliff_for_linear_curve(self):
        curve = 90.0 + 0.03 * np.arange(30)

        sustained = detect_performance_cliff(curve)
        rolling = detect_rolling_sustained_performance_cliff(curve)
        piecewise = detect_piecewise_performance_cliff(curve)
        hybrid = detect_hybrid_performance_cliff(curve)

        self.assertIsNone(sustained["performance_cliff_lap"])
        self.assertIsNone(rolling["performance_cliff_lap"])
        self.assertIsNone(piecewise["performance_cliff_lap"])
        self.assertIsNone(hybrid["performance_cliff_lap"])

    def test_rolling_sustained_detector_finds_persistent_trend_change(self):
        pre = 90.0 + 0.02 * np.arange(12)
        post = pre[-1] + 0.20 * np.arange(1, 19)
        curve = np.concatenate([pre, post])

        result = detect_rolling_sustained_performance_cliff(curve)

        self.assertIn(result["performance_cliff_lap"], range(11, 15))
        self.assertIn(result["cliff_confidence"], {"medium", "high"})
        self.assertEqual(result["cliff_method"], "rolling_sustained")

    def test_rolling_sustained_detector_rejects_late_edge_acceleration(self):
        gradual = 90.0 + 0.03 * np.arange(18)
        unsupported_edge = np.asarray([90.65, 90.90])
        curve = np.concatenate([gradual, unsupported_edge])

        result = detect_rolling_sustained_performance_cliff(curve)

        self.assertIsNone(result["performance_cliff_lap"])

    def test_piecewise_detector_finds_known_breakpoint(self):
        pre = 90.0 + 0.02 * np.arange(12)
        post = pre[-1] + 0.20 * np.arange(1, 19)
        curve = np.concatenate([pre, post])

        result = detect_piecewise_performance_cliff(curve)

        self.assertIn(result["performance_cliff_lap"], range(12, 15))
        self.assertIn(result["cliff_confidence"], {"medium", "high"})
        self.assertEqual(result["cliff_method"], "piecewise")

    def test_hybrid_detector_confirms_persistent_breakpoint(self):
        pre = 90.0 + 0.02 * np.arange(12)
        post = pre[-1] + 0.20 * np.arange(1, 19)
        curve = np.concatenate([pre, post])

        result = detect_hybrid_performance_cliff(curve)

        self.assertIn(result["performance_cliff_lap"], range(12, 15))
        self.assertEqual(result["cliff_method"], "hybrid")

    def test_analyze_tire_life_selects_requested_detector(self):
        pre = 90.0 + 0.02 * np.arange(12)
        post = pre[-1] + 0.20 * np.arange(1, 19)
        curve = np.concatenate([pre, post])

        result = analyze_tire_life(
            curve,
            config={"cliff_detection_method": "piecewise"},
        )

        self.assertEqual(result["cliff_method"], "piecewise")
        self.assertIsNotNone(result["performance_cliff_lap"])

    def test_analyze_tire_life_selects_rolling_sustained_detector(self):
        pre = 90.0 + 0.02 * np.arange(12)
        post = pre[-1] + 0.20 * np.arange(1, 19)
        curve = np.concatenate([pre, post])

        result = analyze_tire_life(
            curve,
            config={"cliff_detection_method": "rolling_sustained"},
        )

        self.assertEqual(result["cliff_method"], "rolling_sustained")
        self.assertIsNotNone(result["performance_cliff_lap"])

    def test_rejects_unknown_detector(self):
        with self.assertRaisesRegex(ValueError, "cliff_detection_method"):
            analyze_tire_life(
                np.linspace(90.0, 91.0, 20),
                config={"cliff_detection_method": "unknown"},
            )

    def test_fixed_fuel_model_curve_does_not_add_fuel_burn_cost(self):
        curve = np.full(30, 90.0)

        observed = recommend_strategy_useful_life(
            curve,
            pit_loss=10.0,
            fuel_correction=True,
        )
        fixed_fuel = recommend_strategy_useful_life(
            curve,
            pit_loss=10.0,
            fuel_correction=False,
        )

        self.assertLess(
            observed["strategy_useful_life_lap"],
            fixed_fuel["strategy_useful_life_lap"],
        )
        self.assertEqual(
            fixed_fuel["strategy_useful_life_lap"],
            29,
        )

    def test_useful_life_uncertainty_is_deterministic_and_bounded(self):
        result_a = estimate_useful_life_uncertainty(
            np.linspace(90.0, 93.0, 30), pit_loss=10.0,
            fuel_correction=False, draws=40, seed=42,
        )
        result_b = estimate_useful_life_uncertainty(
            np.linspace(90.0, 93.0, 30), pit_loss=10.0,
            fuel_correction=False, draws=40, seed=42,
        )
        self.assertEqual(result_a, result_b)
        self.assertIn(result_a["strategy_useful_life_uncertainty_laps"], {1, 2, 3})
        self.assertLessEqual(result_a["strategy_useful_life_lower"], result_a["strategy_useful_life_lap"])
        self.assertLessEqual(result_a["strategy_useful_life_lap"], result_a["strategy_useful_life_upper"])

    def test_no_crossover_receives_low_confidence(self):
        result = estimate_useful_life_uncertainty(
            np.full(30, 90.0), pit_loss=100.0,
            fuel_correction=False, draws=20, seed=42,
        )
        self.assertEqual(result["strategy_useful_life_confidence"], "low")


if __name__ == "__main__":
    unittest.main()
