#!/usr/bin/env python3
"""Diagnose observed-versus-predicted cliff curve shape on calibration data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _slope(ages: np.ndarray, values: np.ndarray) -> float | None:
    if len(ages) < 2:
        return None
    return float(np.polyfit(ages, values, 1)[0])


def _stint_metrics(capture: dict) -> dict:
    observed = {
        int(point["tire_age"]): float(
            point["fuel_corrected_lap_time_seconds"]
        )
        for point in capture["observed_curve"]
    }
    predicted = {
        int(point["tire_age"]): float(
            point["predicted_lap_time_seconds"]
        )
        for point in capture["predicted_curve"]
    }
    ages = sorted(set(observed) & set(predicted))
    if len(ages) < 2:
        raise ValueError(
            f"{capture['reference_id']}: fewer than two aligned ages"
        )
    age_values = np.asarray(ages, dtype=float)
    observed_values = np.asarray(
        [observed[age] for age in ages],
        dtype=float,
    )
    predicted_values = np.asarray(
        [predicted[age] for age in ages],
        dtype=float,
    )
    observed_shape = observed_values - observed_values[0]
    predicted_shape = predicted_values - predicted_values[0]
    result = {
        "reference_id": capture["reference_id"],
        "race_id": capture.get("race_id"),
        "compound": capture["compound"],
        "manual_review_status": capture["manual_review_status"],
        "reviewed_cliff_lap": capture.get("reviewed_cliff_lap"),
        "aligned_age_count": len(ages),
        "starting_tire_age": ages[0],
        "ending_tire_age": ages[-1],
        "observed_end_delta_sec": round(
            float(observed_shape[-1]),
            6,
        ),
        "predicted_end_delta_sec": round(
            float(predicted_shape[-1]),
            6,
        ),
        "normalized_shape_mae_sec": round(
            float(np.mean(np.abs(observed_shape - predicted_shape))),
            6,
        ),
    }
    cliff_lap = capture.get("reviewed_cliff_lap")
    if cliff_lap is not None:
        pre = age_values < int(cliff_lap)
        post = age_values >= int(cliff_lap)
        result.update(
            {
                "observed_pre_cliff_slope_sec_per_lap": _slope(
                    age_values[pre],
                    observed_values[pre],
                ),
                "predicted_pre_cliff_slope_sec_per_lap": _slope(
                    age_values[pre],
                    predicted_values[pre],
                ),
                "observed_post_cliff_slope_sec_per_lap": _slope(
                    age_values[post],
                    observed_values[post],
                ),
                "predicted_post_cliff_slope_sec_per_lap": _slope(
                    age_values[post],
                    predicted_values[post],
                ),
            }
        )
    return result


def _summary(rows: list[dict]) -> dict:
    fields = (
        "observed_end_delta_sec",
        "predicted_end_delta_sec",
        "normalized_shape_mae_sec",
        "observed_pre_cliff_slope_sec_per_lap",
        "predicted_pre_cliff_slope_sec_per_lap",
        "observed_post_cliff_slope_sec_per_lap",
        "predicted_post_cliff_slope_sec_per_lap",
    )
    summary = {"stint_count": len(rows)}
    for field in fields:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None
        ]
        if values:
            summary[f"{field}_mean"] = round(mean(values), 6)
            summary[f"{field}_median"] = round(median(values), 6)
    observed_post = summary.get(
        "observed_post_cliff_slope_sec_per_lap_mean"
    )
    predicted_post = summary.get(
        "predicted_post_cliff_slope_sec_per_lap_mean"
    )
    if observed_post not in (None, 0) and predicted_post is not None:
        summary["post_cliff_slope_retention_ratio"] = round(
            predicted_post / observed_post,
            6,
        )
    return summary


def diagnose(predictions: dict) -> dict:
    if (
        predictions.get("artifact_type")
        != "tire_cliff_calibration_predictions"
    ):
        raise ValueError(
            "expected a tire_cliff_calibration_predictions artifact"
        )
    if predictions.get("split") != "calibration":
        raise ValueError("curve diagnosis accepts calibration only")
    captures = predictions.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("prediction artifact contains no captures")
    if any(capture.get("split") != "calibration" for capture in captures):
        raise ValueError("curve diagnosis cannot read holdout captures")

    rows = [_stint_metrics(capture) for capture in captures]
    by_status = defaultdict(list)
    by_compound = defaultdict(list)
    for row in rows:
        by_status[row["manual_review_status"]].append(row)
        by_compound[row["compound"]].append(row)
    cliff_summary = _summary(by_status.get("confirmed_cliff", []))
    slope_retention = cliff_summary.get(
        "post_cliff_slope_retention_ratio"
    )
    findings = []
    if slope_retention is not None and slope_retention < 0.5:
        findings.append(
            {
                "code": "predicted_post_cliff_slope_attenuation",
                "severity": "material",
                "detail": (
                    "Mean predicted post-cliff slope retains less than "
                    "half of the reviewed observed post-cliff slope."
                ),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "tire_cliff_curve_shape_diagnosis",
        "split": "calibration",
        "capture_count": len(rows),
        "review_artifact_sha256": predictions.get(
            "review_artifact_sha256"
        ),
        "model_provenance": predictions.get("model_provenance"),
        "aggregate": {
            "all": _summary(rows),
            "review_status": {
                key: _summary(values)
                for key, values in sorted(by_status.items())
            },
            "compound": {
                key: _summary(values)
                for key, values in sorted(by_compound.items())
            },
        },
        "findings": findings,
        "stints": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    report = diagnose(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        f"Diagnosed {report['capture_count']} calibration curves; "
        f"findings={len(report['findings'])}; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
