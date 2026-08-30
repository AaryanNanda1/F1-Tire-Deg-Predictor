#!/usr/bin/env python3
"""Run a bounded cliff-detector search on frozen calibration captures only.

This script produces a diagnostic artifact. It never reads holdout data,
changes production defaults, trains a model, commits, pushes, or deploys.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_tire_cliffs import (  # noqa: E402
    _captured_curve,
    _observed_curve,
    _raw_captured_curve,
    _score_rows,
)
from tire_life_analysis import (  # noqa: E402
    DEFAULT_CONFIG,
    detect_hybrid_performance_cliff,
    detect_performance_cliff,
    detect_piecewise_performance_cliff,
    detect_rolling_sustained_performance_cliff,
    smooth_lap_times,
)


FALSE_CLIFF_GUARDRAIL = 0.30
MINIMUM_CLIFF_RECALL = 0.50
TOP_CANDIDATE_COUNT = 10


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configs(keys, values):
    return [
        dict(zip(keys, combination))
        for combination in product(*values)
    ]


def parameter_grids() -> dict[str, list[dict]]:
    """Return deterministic, deliberately bounded candidate grids."""
    sustained = _configs(
        (
            "cliff_slope_threshold",
            "cliff_curvature_threshold",
            "cliff_baseline_delta",
            "cliff_persistence_laps",
            "cliff_min_lap",
        ),
        (
            (0.02, 0.03, 0.05, 0.08),
            (0.0, 0.002, 0.005, 0.01),
            (0.15, 0.3, 0.5),
            (1, 2, 3),
            (4, 5, 7),
        ),
    )
    piecewise = _configs(
        (
            "piecewise_min_segment_laps",
            "cliff_slope_threshold",
            "piecewise_min_slope_increase",
            "piecewise_min_improvement_ratio",
        ),
        (
            (3, 4, 5),
            (0.02, 0.05, 0.08),
            (0.03, 0.05, 0.08, 0.12),
            (0.1, 0.2, 0.3),
        ),
    )
    hybrid = _configs(
        (
            "piecewise_min_segment_laps",
            "cliff_slope_threshold",
            "piecewise_min_slope_increase",
            "piecewise_min_improvement_ratio",
            "cliff_persistence_laps",
            "cliff_baseline_delta",
        ),
        (
            (3, 4, 5),
            (0.02, 0.05, 0.08),
            (0.03, 0.08, 0.12),
            (0.1, 0.2, 0.3),
            (1, 2),
            (0.15, 0.3),
        ),
    )
    rolling_sustained = _configs(
        (
            "rolling_trend_window",
            "cliff_slope_threshold",
            "rolling_min_slope_increase",
            "cliff_baseline_delta",
            "rolling_min_fit_improvement_ratio",
            "cliff_persistence_laps",
        ),
        (
            (3, 4, 5),
            (0.03, 0.05, 0.08),
            (0.03, 0.05, 0.08, 0.12),
            (0.15, 0.3, 0.5),
            (0.1, 0.2, 0.3),
            (1, 2),
        ),
    )
    return {
        "sustained": sustained,
        "rolling_sustained": rolling_sustained,
        "piecewise": piecewise,
        "hybrid": hybrid,
    }


def _objective(metrics: dict) -> dict:
    """Balance classification first, then timing and early-cliff bias."""
    balanced_accuracy = metrics["balanced_accuracy"] or 0.0
    mae = metrics["mean_absolute_cliff_lap_error"]
    timing_loss = 1.0 if mae is None else min(float(mae) / 10.0, 1.0)
    bias = metrics["mean_cliff_lap_error_bias"]
    early_bias_loss = (
        0.0
        if bias is None
        else min(max(0.0, -float(bias)) / 10.0, 1.0)
    )
    objective_loss = (
        (1.0 - balanced_accuracy)
        + 0.15 * timing_loss
        + 0.05 * early_bias_loss
    )
    return {
        "objective_loss": round(objective_loss, 6),
        "classification_loss": round(1.0 - balanced_accuracy, 6),
        "timing_loss": round(timing_loss, 6),
        "early_bias_loss": round(early_bias_loss, 6),
    }


def _prepare_captures(
    captures: list[dict],
    *,
    curve_source: str,
) -> list[dict]:
    prepared = []
    curve_loader = {
        "predicted": _captured_curve,
        "raw_predicted": _raw_captured_curve,
        "observed": _observed_curve,
    }.get(curve_source)
    if curve_loader is None:
        raise ValueError(
            "curve_source must be predicted, raw_predicted, or observed"
        )
    for capture in captures:
        curve, tire_age_offset = curve_loader(capture)
        prepared.append(
            {
                "capture": capture,
                "smoothed_curve": smooth_lap_times(curve),
                "tire_age_offset": tire_age_offset,
            }
        )
    return prepared


def _score_config(
    prepared_captures: list[dict],
    detector,
    config: dict,
) -> dict:
    rows = []
    for prepared in prepared_captures:
        capture = prepared["capture"]
        result = detector(prepared["smoothed_curve"], config)
        relative_lap = result.get("performance_cliff_lap")
        detected_lap = (
            int(relative_lap + prepared["tire_age_offset"])
            if relative_lap is not None
            else None
        )
        start = int(capture["starting_tire_age"])
        end = int(capture["ending_tire_age"])
        in_window = (
            detected_lap is not None and start <= detected_lap <= end
        )
        rows.append(
            {
                "manual_review_status": capture["manual_review_status"],
                "reviewed_cliff_lap": capture.get("reviewed_cliff_lap"),
                "detected_cliff_lap": detected_lap,
                "detected_in_review_window": in_window,
                "predicted_outside_review_window": (
                    detected_lap is not None and not in_window
                ),
            }
        )
    return _score_rows(rows)


def _rank_key(candidate: dict):
    metrics = candidate["metrics"]
    return (
        candidate["objective"]["objective_loss"],
        metrics["false_cliff_rate"]
        if metrics["false_cliff_rate"] is not None
        else 1.0,
        metrics["missed_cliff_rate"]
        if metrics["missed_cliff_rate"] is not None
        else 1.0,
        metrics["mean_absolute_cliff_lap_error"]
        if metrics["mean_absolute_cliff_lap_error"] is not None
        else float("inf"),
        json.dumps(candidate["config"], sort_keys=True),
    )


def search_method(
    prepared_captures: list[dict],
    detector,
    configs: list[dict],
) -> dict:
    candidates = []
    for config in configs:
        metrics = _score_config(prepared_captures, detector, config)
        candidate = {
            "config": config,
            "metrics": metrics,
            "objective": _objective(metrics),
            "passes_acceptance_guardrails": (
                metrics["false_cliff_rate"] is not None
                and metrics["false_cliff_rate"]
                <= FALSE_CLIFF_GUARDRAIL
                and metrics["recall"] is not None
                and metrics["recall"] >= MINIMUM_CLIFF_RECALL
                and metrics["true_positive_count"] > 0
            ),
        }
        candidates.append(candidate)
    candidates.sort(key=_rank_key)
    eligible = [
        candidate
        for candidate in candidates
        if candidate["passes_acceptance_guardrails"]
    ]
    return {
        "searched_config_count": len(candidates),
        "eligible_config_count": len(eligible),
        "best_overall": candidates[0],
        "best_guardrail_eligible": eligible[0] if eligible else None,
        "top_candidates": candidates[:TOP_CANDIDATE_COUNT],
    }


def _default_method_config(method: str) -> dict:
    keys = {
        "sustained": (
            "cliff_slope_threshold",
            "cliff_curvature_threshold",
            "cliff_baseline_delta",
            "cliff_persistence_laps",
            "cliff_min_lap",
        ),
        "rolling_sustained": (
            "rolling_trend_window",
            "cliff_slope_threshold",
            "rolling_min_slope_increase",
            "cliff_baseline_delta",
            "rolling_min_fit_improvement_ratio",
            "cliff_persistence_laps",
        ),
        "piecewise": (
            "piecewise_min_segment_laps",
            "cliff_slope_threshold",
            "piecewise_min_slope_increase",
            "piecewise_min_improvement_ratio",
        ),
        "hybrid": (
            "piecewise_min_segment_laps",
            "cliff_slope_threshold",
            "piecewise_min_slope_increase",
            "piecewise_min_improvement_ratio",
            "cliff_persistence_laps",
            "cliff_baseline_delta",
        ),
    }[method]
    return {key: DEFAULT_CONFIG[key] for key in keys}


def run_search(
    predictions: dict,
    *,
    grids=None,
    curve_source: str = "predicted",
    method_names: tuple[str, ...] | None = None,
) -> dict:
    if (
        predictions.get("artifact_type")
        != "tire_cliff_calibration_predictions"
    ):
        raise ValueError(
            "expected a tire_cliff_calibration_predictions artifact"
        )
    if predictions.get("split") != "calibration":
        raise ValueError("parameter tuning accepts calibration only")
    captures = predictions.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("prediction artifact contains no captures")
    if any(capture.get("split") != "calibration" for capture in captures):
        raise ValueError("parameter tuning cannot read holdout captures")

    prepared = _prepare_captures(
        captures,
        curve_source=curve_source,
    )
    grids = grids or parameter_grids()
    detectors = {
        "sustained": detect_performance_cliff,
        "rolling_sustained": (
            detect_rolling_sustained_performance_cliff
        ),
        "piecewise": detect_piecewise_performance_cliff,
        "hybrid": detect_hybrid_performance_cliff,
    }
    method_names = method_names or tuple(detectors)
    unknown_methods = sorted(set(method_names) - set(detectors))
    if unknown_methods:
        raise ValueError(
            f"unknown cliff detector methods: {unknown_methods}"
        )
    methods = {}
    for name in method_names:
        detector = detectors[name]
        default_config = _default_method_config(name)
        methods[name] = {
            "default": {
                "config": default_config,
                "metrics": _score_config(
                    prepared,
                    detector,
                    default_config,
                ),
            },
            "search": search_method(
                prepared,
                detector,
                grids[name],
            ),
        }
        methods[name]["default"]["objective"] = _objective(
            methods[name]["default"]["metrics"]
        )

    eligible = [
        {
            "method": name,
            **method["search"]["best_guardrail_eligible"],
        }
        for name, method in methods.items()
        if method["search"]["best_guardrail_eligible"] is not None
    ]
    eligible.sort(key=_rank_key)
    return {
        "schema_version": 1,
        "artifact_type": "tire_cliff_calibration_parameter_search",
        "split": "calibration",
        "curve_source": curve_source,
        "capture_count": len(captures),
        "review_artifact_sha256": predictions.get(
            "review_artifact_sha256"
        ),
        "model_provenance": predictions.get("model_provenance"),
        "objective": {
            "formula": (
                "(1 - balanced_accuracy) + 0.15 * "
                "min(matched_cliff_mae / 10, 1) + 0.05 * "
                "min(max(0, -timing_bias) / 10, 1)"
            ),
            "false_cliff_guardrail": FALSE_CLIFF_GUARDRAIL,
            "minimum_cliff_recall": MINIMUM_CLIFF_RECALL,
            "guardrail_note": (
                "Eligible candidates must meet the minimum confirmed-cliff "
                "recall and keep the confirmed-no-cliff false-cliff rate "
                "at or below the guardrail."
            ),
        },
        "methods": methods,
        "best_guardrail_eligible": eligible[0] if eligible else None,
        "selection_warning": (
            "This search does not change production defaults. Calibration "
            "results must be diagnosed and frozen before holdout evaluation. "
            + (
                "Observed curves use linear interpolation only across tire "
                "ages removed by the frozen cleaning policy."
                if curve_source == "observed"
                else "Predicted curves are evaluated through each reviewed "
                "stint endpoint."
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--curve-source",
        choices=("predicted", "raw_predicted", "observed"),
        default="predicted",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=(
            "sustained",
            "rolling_sustained",
            "piecewise",
            "hybrid",
        ),
        dest="methods",
        help=(
            "Limit the search to one or more detector methods. "
            "May be supplied multiple times."
        ),
    )
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    report = run_search(
        predictions,
        curve_source=args.curve_source,
        method_names=tuple(args.methods) if args.methods else None,
    )
    report["prediction_artifact"] = str(args.predictions)
    report["prediction_artifact_sha256"] = _sha256(args.predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        "Completed bounded calibration search: "
        f"source={args.curve_source}; "
        + ", ".join(
            f"{name}={details['search']['searched_config_count']}"
            for name, details in report["methods"].items()
        )
        + f"; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
