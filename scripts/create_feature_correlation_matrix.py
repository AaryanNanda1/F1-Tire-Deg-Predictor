#!/usr/bin/env python3
"""Create a correlation heat map for the current numeric model predictors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_PREDICTORS = [
    "LapNumber",
    "TyreLife",
    "TyreLifeKM",
    "TyreLifeSquared",
    "FuelLoad",
    "FuelLoadMissing",
    "Stint",
    "IsWet",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Rainfall",
    "WindSpeed",
    "traction",
    "tyre_stress",
    "asphalt_grip",
    "corner_speed_energy",
    "abrasiveness",
    "braking_severity",
    "lateral_load",
    "NormalizedLap",
    "LapsRemaining",
    "TireAgeRatio",
    "tire_age_x_abrasiveness",
    "track_temp_x_tyre_stress",
    "tire_age_x_traction",
    "tire_age_x_lateral_load",
    "soft_age_interaction",
    "medium_age_interaction",
    "hard_age_interaction",
    "soft_abrasiveness_interaction",
    "soft_traction_interaction",
]


def load_store(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path, compression="gzip") for path in root.rglob("*.csv.gz")]
    if not frames:
        raise ValueError(f"No processed session files found in {root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def create_heatmap(frame: pd.DataFrame, title: str, output: Path) -> None:
    missing = [column for column in NUMERIC_PREDICTORS if column not in frame]
    if missing:
        raise ValueError(f"Missing expected predictors: {missing}")

    correlations = frame[NUMERIC_PREDICTORS].corr(method="spearman")
    output.parent.mkdir(parents=True, exist_ok=True)

    size = max(16, len(NUMERIC_PREDICTORS) * 0.52)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(
        correlations.to_numpy(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    axis.set_xticks(range(len(NUMERIC_PREDICTORS)), NUMERIC_PREDICTORS)
    axis.set_yticks(range(len(NUMERIC_PREDICTORS)), NUMERIC_PREDICTORS)
    axis.set_title(title, pad=18)
    axis.tick_params(axis="x", rotation=75, labelsize=8)
    axis.tick_params(axis="y", rotation=0, labelsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-effect-dir", default="training_data/ground_effect")
    parser.add_argument("--active-aero-dir", default="training_data/active_aero")
    parser.add_argument(
        "--output",
        default="docs/feature_correlation_matrix.png",
        help="Output PNG path for the combined matrix",
    )
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory for the era-specific matrices",
    )
    args = parser.parse_args()

    ground_effect = load_store(Path(args.ground_effect_dir))
    active_aero = load_store(Path(args.active_aero_dir))
    frame = pd.concat(
        [ground_effect, active_aero],
        ignore_index=True,
        sort=False,
    )
    output_dir = Path(args.output_dir)
    create_heatmap(
        ground_effect,
        "Ground Effect era\nNumeric model predictor correlations",
        output_dir / "ground_effect_feature_correlation_matrix.png",
    )
    create_heatmap(
        active_aero,
        "Active Aero era\nNumeric model predictor correlations",
        output_dir / "active_aero_feature_correlation_matrix.png",
    )
    create_heatmap(
        frame,
        "Numeric model predictor correlations\n"
        "Ground Effect + Active Aero processed stores",
        Path(args.output),
    )

    print(f"Rows analyzed: {len(frame):,}")
    print(f"Ground Effect rows: {len(ground_effect):,}")
    print(f"Active Aero rows: {len(active_aero):,}")
    print(f"Predictors analyzed: {len(NUMERIC_PREDICTORS)}")
    print(f"Saved: {output_dir / 'ground_effect_feature_correlation_matrix.png'}")
    print(f"Saved: {output_dir / 'active_aero_feature_correlation_matrix.png'}")
    print(f"Saved: {Path(args.output)}")


if __name__ == "__main__":
    main()
