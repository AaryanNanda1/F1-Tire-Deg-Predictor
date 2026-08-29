"""Source-backed circuit characteristic feature construction.

The raw observations live in ``data/track_characteristics_2025.csv``.  Keeping
the published ratings and corner speeds outside the model code makes the
normalization reproducible and gives each circuit an auditable source trail.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


DATA_PATH = Path(__file__).resolve().parent / "data" / "track_characteristics_2025.csv"

PIRELLI_RATING_MIN = 1
PIRELLI_RATING_MAX = 5
CORNER_SPEED_REFERENCE_KMH = 300.0

TRACK_FEATURE_NAMES = (
    "traction",
    "tyre_stress",
    "asphalt_grip",
    "braking_severity",
    "abrasiveness",
    "lateral_load",
    "corner_speed_energy",
)

_RATING_COLUMNS = {
    "traction": "pirelli_traction",
    "tyre_stress": "pirelli_tyre_stress",
    "asphalt_grip": "pirelli_asphalt_grip",
    "braking_severity": "pirelli_braking",
    "abrasiveness": "pirelli_asphalt_abrasion",
    "lateral_load": "pirelli_lateral",
}

# FastF1 used both event labels while the physical Barcelona circuit is the
# same source row in this catalogue.
_CIRCUIT_ALIASES = {
    "Circuit de Barcelona-Catalunya (Barcelona, Spain)":
        "Circuit de Barcelona-Catalunya (Spain)",
}


def normalize_pirelli_rating(rating: int | float) -> float:
    """Map a published Pirelli 1--5 rating linearly onto [0, 1]."""
    value = float(rating)
    if not PIRELLI_RATING_MIN <= value <= PIRELLI_RATING_MAX:
        raise ValueError(f"Pirelli rating must be in [1, 5], got {rating!r}")
    return (value - PIRELLI_RATING_MIN) / (
        PIRELLI_RATING_MAX - PIRELLI_RATING_MIN
    )


def compute_corner_speed_energy(
    speeds_kmh: Iterable[int | float],
    reference_kmh: float = CORNER_SPEED_REFERENCE_KMH,
) -> float:
    """Return the mean squared, capped Mercedes minimum-corner-speed score.

    Squaring is an energy-like weighting that gives fast corners more influence.
    It is a circuit descriptor, not an estimate of tire force or vehicle energy.
    """
    speeds = [float(speed) for speed in speeds_kmh]
    if not speeds:
        raise ValueError("At least one corner speed is required")
    if reference_kmh <= 0:
        raise ValueError("reference_kmh must be positive")
    if any(speed < 0 for speed in speeds):
        raise ValueError("Corner speeds cannot be negative")

    return sum(
        (min(speed, reference_kmh) / reference_kmh) ** 2
        for speed in speeds
    ) / len(speeds)


def parse_turn_speed_pairs(value: str) -> dict[int, int]:
    """Parse ``turn:speed`` pairs from the source CSV."""
    pairs: dict[int, int] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        turn_text, speed_text = item.split(":", maxsplit=1)
        turn = int(turn_text)
        speed = int(speed_text)
        if turn in pairs:
            raise ValueError(f"Duplicate turn {turn} in corner-speed data")
        pairs[turn] = speed
    if not pairs:
        raise ValueError("No Mercedes corner speeds found")
    return pairs


def _parse_missing_turns(value: str) -> tuple[int, ...]:
    return tuple(int(turn) for turn in value.split(";") if turn.strip())


def load_track_characteristics(
    data_path: str | Path = DATA_PATH,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, object]]]:
    """Load raw official observations and derive normalized circuit features."""
    features_by_circuit: dict[str, dict[str, float]] = {}
    sources_by_circuit: dict[str, dict[str, object]] = {}

    with Path(data_path).open(newline="", encoding="utf-8") as source_file:
        for row in csv.DictReader(source_file):
            circuit = row["circuit"]
            if circuit in features_by_circuit:
                raise ValueError(f"Duplicate circuit in source data: {circuit}")

            turn_speeds = parse_turn_speed_pairs(row["mercedes_turn_speeds_kmh"])
            normalized = {
                feature: normalize_pirelli_rating(row[column])
                for feature, column in _RATING_COLUMNS.items()
            }
            normalized["corner_speed_energy"] = compute_corner_speed_energy(
                turn_speeds.values()
            )

            features_by_circuit[circuit] = normalized
            sources_by_circuit[circuit] = {
                "reference_season": int(row["reference_season"]),
                "pirelli_event_code": row["pirelli_event_code"],
                "pirelli_article_url": row["pirelli_article_url"],
                "pirelli_graphic_url": row["pirelli_graphic_url"],
                "pirelli_ratings": {
                    feature: int(row[column])
                    for feature, column in _RATING_COLUMNS.items()
                },
                "mercedes_source_year": int(row["mercedes_source_year"]),
                "mercedes_asset_id": row["mercedes_asset_id"],
                "mercedes_page_url": row["mercedes_page_url"],
                "mercedes_turn_speeds_kmh": turn_speeds,
                "mercedes_missing_turns": _parse_missing_turns(
                    row["mercedes_missing_turns"]
                ),
                "notes": row["notes"],
            }

    for alias, canonical in _CIRCUIT_ALIASES.items():
        features_by_circuit[alias] = features_by_circuit[canonical].copy()
        sources_by_circuit[alias] = sources_by_circuit[canonical].copy()

    return features_by_circuit, sources_by_circuit


TRACK_CHARACTERISTICS, TRACK_CHARACTERISTIC_SOURCES = load_track_characteristics()
