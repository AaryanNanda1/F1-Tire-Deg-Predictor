#!/usr/bin/env python3
"""Evaluate tire-cliff detectors on captured prediction curves.

Two input artifacts are supported:

* Pirelli prediction suites receive a curve-only detector comparison.
* Frozen FastF1 calibration captures receive classification and cliff-timing
  metrics. These are calibration results, not untouched holdout results.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tire_life_analysis import (  # noqa: E402
    detect_hybrid_performance_cliff,
    detect_performance_cliff,
    detect_piecewise_performance_cliff,
    detect_rolling_sustained_performance_cliff,
    smooth_lap_times,
)


DETECTORS = (
    ("sustained", detect_performance_cliff),
    (
        "rolling_sustained",
        detect_rolling_sustained_performance_cliff,
    ),
    ("piecewise", detect_piecewise_performance_cliff),
    ("hybrid", detect_hybrid_performance_cliff),
)
USABLE_REVIEW_STATUSES = {"confirmed_cliff", "confirmed_no_cliff"}


def _ordered_curve(values):
    return [
        float(value)
        for _, value in sorted(
            values.items(),
            key=lambda item: int(item[0]),
        )
    ]


def compare_prediction_suite(prediction_suite):
    """Retain the original curve-only comparison for Pirelli captures."""
    comparisons = {}
    for race_id, race_output in prediction_suite.get("races", {}).items():
        compounds = race_output.get("degradation_graphs", {})
        race_comparisons = {}
        for compound, values in compounds.items():
            graph = values.get("smoothed_graph_data") or values.get(
                "graph_data"
            )
            if not graph:
                continue
            curve = _ordered_curve(graph)
            race_comparisons[compound] = {
                "production_cliff_lap": values.get(
                    "performance_cliff_lap"
                ),
                "production_cliff_method": values.get("cliff_method"),
                "strategy_useful_life_lap": values.get(
                    "strategy_useful_life_lap"
                ),
                "detectors": {
                    name: detector(curve)
                    for name, detector in DETECTORS
                },
            }
        comparisons[race_id] = race_comparisons

    return {
        "schema_version": 1,
        "evaluation_mode": "detector_comparison_diagnostic",
        "accuracy_warning": (
            "Detector outputs are not accuracy scores until compared with "
            "reviewed FastF1 cliff references."
        ),
        "races": comparisons,
    }


def _captured_curve(capture: dict) -> tuple[np.ndarray, int]:
    points = capture.get("full_predicted_curve") or capture.get(
        "predicted_curve"
    )
    if not isinstance(points, list) or not points:
        raise ValueError(
            f"{capture.get('reference_id')}: missing predicted curve"
        )
    ordered = sorted(points, key=lambda point: int(point["tire_age"]))
    ages = [int(point["tire_age"]) for point in ordered]
    expected = list(range(ages[0], ages[-1] + 1))
    if ages != expected:
        raise ValueError(
            f"{capture.get('reference_id')}: predicted tire ages must be "
            "continuous"
        )
    curve = np.asarray(
        [
            float(point["predicted_lap_time_seconds"])
            for point in ordered
        ],
        dtype=float,
    )
    return curve, ages[0] - 1


def _raw_captured_curve(capture: dict) -> tuple[np.ndarray, int]:
    """Load the pre-monotonic degradation-model signal."""
    points = capture.get("full_predicted_curve") or capture.get(
        "predicted_curve"
    )
    if not isinstance(points, list) or not points:
        raise ValueError(
            f"{capture.get('reference_id')}: missing predicted curve"
        )
    ordered = sorted(points, key=lambda point: int(point["tire_age"]))
    ages = [int(point["tire_age"]) for point in ordered]
    expected = list(range(ages[0], ages[-1] + 1))
    if ages != expected:
        raise ValueError(
            f"{capture.get('reference_id')}: raw predicted tire ages must "
            "be continuous"
        )
    if any(
        point.get("raw_predicted_drop_off_seconds") is None
        for point in ordered
    ):
        raise ValueError(
            f"{capture.get('reference_id')}: missing raw predicted drop-off"
        )
    curve = np.asarray(
        [
            float(point["raw_predicted_drop_off_seconds"])
            for point in ordered
        ],
        dtype=float,
    )
    return curve, ages[0] - 1


def _observed_curve(capture: dict) -> tuple[np.ndarray, int]:
    """Interpolate cleaned observations onto consecutive integer tire ages."""
    points = capture.get("observed_curve")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(
            f"{capture.get('reference_id')}: insufficient observed curve"
        )
    ordered = sorted(points, key=lambda point: int(point["tire_age"]))
    ages = np.asarray(
        [int(point["tire_age"]) for point in ordered],
        dtype=int,
    )
    if len(set(ages.tolist())) != len(ages):
        raise ValueError(
            f"{capture.get('reference_id')}: duplicate observed tire ages"
        )
    values = np.asarray(
        [
            float(point["fuel_corrected_lap_time_seconds"])
            for point in ordered
        ],
        dtype=float,
    )
    consecutive_ages = np.arange(ages[0], ages[-1] + 1, dtype=int)
    interpolated = np.interp(consecutive_ages, ages, values)
    return interpolated, int(consecutive_ages[0] - 1)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _rounded_mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def _rounded_median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def _classification(row: dict) -> str:
    truth_is_cliff = row["manual_review_status"] == "confirmed_cliff"
    detected = row["detected_in_review_window"]
    if truth_is_cliff:
        return "true_positive" if detected else "false_negative"
    return "false_positive" if detected else "true_negative"


def _score_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    positives = sum(
        row["manual_review_status"] == "confirmed_cliff" for row in rows
    )
    negatives = len(rows) - positives
    classifications = [_classification(row) for row in rows]
    true_positives = classifications.count("true_positive")
    false_negatives = classifications.count("false_negative")
    true_negatives = classifications.count("true_negative")
    false_positives = classifications.count("false_positive")
    errors = [
        float(row["detected_cliff_lap"] - row["reviewed_cliff_lap"])
        for row, classification in zip(rows, classifications)
        if classification == "true_positive"
    ]
    absolute_errors = [abs(value) for value in errors]
    early = sum(value < 0 for value in errors)
    late = sum(value > 0 for value in errors)
    exact = sum(value == 0 for value in errors)
    precision_denominator = true_positives + false_positives
    recall = _rate(true_positives, positives)
    specificity = _rate(true_negatives, negatives)
    precision = _rate(true_positives, precision_denominator)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 6)
    if recall is None or specificity is None:
        balanced_accuracy = None
    else:
        balanced_accuracy = round((recall + specificity) / 2, 6)

    return {
        "evaluated_stint_count": len(rows),
        "confirmed_cliff_count": positives,
        "confirmed_no_cliff_count": negatives,
        "true_positive_count": true_positives,
        "false_negative_count": false_negatives,
        "true_negative_count": true_negatives,
        "false_positive_count": false_positives,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "missed_cliff_rate": _rate(false_negatives, positives),
        "false_cliff_rate": _rate(false_positives, negatives),
        "matched_cliff_count": len(errors),
        "mean_absolute_cliff_lap_error": _rounded_mean(absolute_errors),
        "median_absolute_cliff_lap_error": _rounded_median(
            absolute_errors
        ),
        "mean_cliff_lap_error_bias": _rounded_mean(errors),
        "early_detection_count": early,
        "late_detection_count": late,
        "exact_detection_count": exact,
        "early_detection_rate_among_matched": _rate(early, len(errors)),
        "late_detection_rate_among_matched": _rate(late, len(errors)),
        "within_1_lap_rate_among_matched": _rate(
            sum(value <= 1 for value in absolute_errors),
            len(errors),
        ),
        "within_3_laps_rate_among_matched": _rate(
            sum(value <= 3 for value in absolute_errors),
            len(errors),
        ),
        "outside_review_window_prediction_count": sum(
            row["predicted_outside_review_window"] for row in rows
        ),
    }


def _temperature_band(row: dict) -> str:
    temperature = row.get("mean_track_temp_c")
    if temperature is None:
        return "unknown"
    if temperature < 30:
        return "cool_lt_30c"
    if temperature < 40:
        return "moderate_30_to_39c"
    return "hot_ge_40c"


def _segment_scores(
    rows: list[dict],
    key: Callable[[dict], str],
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(key(row))].append(row)
    return {
        segment: _score_rows(grouped[segment])
        for segment in sorted(grouped)
    }


def _evaluate_detector(
    captures: list[dict],
    detector: Callable[[np.ndarray], dict],
) -> dict:
    rows = []
    for capture in captures:
        curve, tire_age_offset = _captured_curve(capture)
        result = detector(smooth_lap_times(curve))
        relative_cliff_lap = result.get("performance_cliff_lap")
        detected_cliff_lap = (
            int(relative_cliff_lap + tire_age_offset)
            if relative_cliff_lap is not None
            else None
        )
        start = int(capture["starting_tire_age"])
        end = int(capture["ending_tire_age"])
        detected_in_window = (
            detected_cliff_lap is not None
            and start <= detected_cliff_lap <= end
        )
        row = {
            "reference_id": capture["reference_id"],
            "race_id": capture.get(
                "race_id",
                f"{capture['season']}:{capture['event_name']}",
            ),
            "season": int(capture["season"]),
            "event_name": capture["event_name"],
            "track_name": capture["track_name"],
            "track_type": capture.get("track_type", "unknown"),
            "driver": capture["driver"],
            "team": capture["team"],
            "compound": capture["compound"],
            "starting_tire_age": start,
            "ending_tire_age": end,
            "manual_review_status": capture["manual_review_status"],
            "reviewed_cliff_lap": capture.get("reviewed_cliff_lap"),
            "detected_cliff_lap": detected_cliff_lap,
            "detected_in_review_window": detected_in_window,
            "predicted_outside_review_window": (
                detected_cliff_lap is not None and not detected_in_window
            ),
            "cliff_confidence": result.get("cliff_confidence"),
            "cliff_reason": result.get("cliff_reason"),
            "mean_track_temp_c": capture.get(
                "weather_context",
                {},
            ).get("track_temp"),
            "training_overlap": bool(capture.get("training_overlap")),
        }
        row["classification"] = _classification(row)
        row["cliff_lap_error"] = (
            detected_cliff_lap - row["reviewed_cliff_lap"]
            if row["classification"] == "true_positive"
            else None
        )
        rows.append(row)

    return {
        "overall": _score_rows(rows),
        "segments": {
            "compound": _segment_scores(
                rows,
                lambda row: row["compound"],
            ),
            "race": _segment_scores(rows, lambda row: row["race_id"]),
            "track_type": _segment_scores(
                rows,
                lambda row: row["track_type"],
            ),
            "track_temperature": _segment_scores(
                rows,
                _temperature_band,
            ),
            "training_overlap": _segment_scores(
                rows,
                lambda row: (
                    "overlap" if row["training_overlap"] else "no_overlap"
                ),
            ),
            "predicted_confidence": _segment_scores(
                rows,
                lambda row: row["cliff_confidence"] or "no_cliff",
            ),
        },
        "stints": rows,
    }


def evaluate_calibration_predictions(
    prediction_suite: dict,
    *,
    detectors=DETECTORS,
) -> dict:
    """Score candidate detectors against frozen calibration labels."""
    if (
        prediction_suite.get("artifact_type")
        != "tire_cliff_calibration_predictions"
    ):
        raise ValueError(
            "expected a tire_cliff_calibration_predictions artifact"
        )
    if prediction_suite.get("split") != "calibration":
        raise ValueError("cliff detector calibration accepts calibration only")
    captures = prediction_suite.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("prediction artifact contains no captures")
    for capture in captures:
        reference_id = capture.get("reference_id", "<unknown>")
        if capture.get("split") != "calibration":
            raise ValueError(f"{reference_id}: non-calibration capture")
        status = capture.get("manual_review_status")
        if status not in USABLE_REVIEW_STATUSES:
            raise ValueError(
                f"{reference_id}: unsupported review status {status!r}"
            )
        if status == "confirmed_cliff":
            cliff_lap = capture.get("reviewed_cliff_lap")
            if cliff_lap is None:
                raise ValueError(
                    f"{reference_id}: confirmed cliff requires a lap"
                )

    overlap_count = sum(bool(item.get("training_overlap")) for item in captures)
    return {
        "schema_version": 1,
        "evaluation_mode": "frozen_calibration_label_evaluation",
        "split": "calibration",
        "accuracy_scope": (
            "These metrics support detector calibration only. They are not "
            "an untouched holdout or generalization score."
        ),
        "review_artifact": prediction_suite.get("review_artifact"),
        "review_artifact_sha256": prediction_suite.get(
            "review_artifact_sha256"
        ),
        "model_provenance": prediction_suite.get("model_provenance"),
        "capture_count": len(captures),
        "training_overlap_capture_count": overlap_count,
        "detectors": {
            name: _evaluate_detector(captures, detector)
            for name, detector in detectors
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8") as handle:
        prediction_suite = json.load(handle)
    if (
        prediction_suite.get("artifact_type")
        == "tire_cliff_calibration_predictions"
    ):
        report = evaluate_calibration_predictions(prediction_suite)
    else:
        report = compare_prediction_suite(prediction_suite)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
