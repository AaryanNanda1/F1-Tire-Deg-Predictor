"""Prioritize and validate manually reviewed tire-cliff calibration labels."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


REVIEW_STATUSES = {
    "pending",
    "confirmed_cliff",
    "confirmed_no_cliff",
    "rejected",
}


class CliffReviewValidationError(ValueError):
    """Raised when a review queue or completed label is invalid."""


def _quality(stint: Mapping[str, Any]) -> float:
    clean_laps = float(stint.get("clean_lap_count") or 0)
    removed_fraction = float(stint.get("removed_fraction") or 0)
    return clean_laps * max(0.0, 1.0 - removed_fraction)


def _balanced_sample(
    candidates: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[Mapping[str, Any]]:
    remaining = list(candidates)
    selected: list[Mapping[str, Any]] = []
    race_counts: Counter[str] = Counter()
    compound_counts: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        best = min(
            remaining,
            key=lambda stint: (
                compound_counts[str(stint.get("compound"))],
                race_counts[str(stint.get("race_id"))],
                -_quality(stint),
                str(stint.get("reference_id")),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        race_counts[str(best.get("race_id"))] += 1
        compound_counts[str(best.get("compound"))] += 1
    return selected


def build_review_queue(
    references: Mapping[str, Any],
    *,
    medium_limit: int = 24,
    no_cliff_limit: int = 36,
) -> list[dict[str, Any]]:
    """Build a deterministic, calibration-only balanced review queue."""
    if references.get("schema_version") != 1:
        raise CliffReviewValidationError("references schema_version must be 1")
    stints = references.get("accepted_stints")
    if not isinstance(stints, list) or not stints:
        raise CliffReviewValidationError("references accepted_stints must be non-empty")
    if medium_limit < 0 or no_cliff_limit < 0:
        raise CliffReviewValidationError("review limits cannot be negative")
    if any(stint.get("split") != "calibration" for stint in stints):
        raise CliffReviewValidationError(
            "review prioritization accepts calibration stints only"
        )

    ids = [str(stint.get("reference_id") or "") for stint in stints]
    if any(not reference_id for reference_id in ids):
        raise CliffReviewValidationError("every stint requires reference_id")
    if len(set(ids)) != len(ids):
        raise CliffReviewValidationError("reference_id values must be unique")

    high = [
        stint
        for stint in stints
        if stint.get("reference_status") == "candidate_for_manual_review"
        and stint.get("reference_confidence") == "high"
    ]
    medium = [
        stint
        for stint in stints
        if stint.get("reference_status") == "candidate_for_manual_review"
        and stint.get("reference_confidence") != "high"
    ]
    no_cliff = [
        stint
        for stint in stints
        if stint.get("reference_status") == "no_cliff_candidate"
    ]

    selected = [
        *sorted(high, key=lambda stint: (-_quality(stint), stint["reference_id"])),
        *_balanced_sample(medium, medium_limit),
        *_balanced_sample(no_cliff, no_cliff_limit),
    ]
    high_ids = {stint["reference_id"] for stint in high}
    medium_ids = {stint["reference_id"] for stint in medium}
    queue = []
    for review_order, stint in enumerate(selected, start=1):
        reference_id = stint["reference_id"]
        if reference_id in high_ids:
            reason = "required_high_confidence_candidate"
        elif reference_id in medium_ids:
            reason = "balanced_medium_confidence_sample"
        else:
            reason = "balanced_no_cliff_sample"
        queue.append(
            {
                **stint,
                "review_order": review_order,
                "review_priority_reason": reason,
                "manual_review_status": "pending",
                "reviewed_cliff_lap": None,
                "review_notes": None,
            }
        )
    return queue


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and not isfinite(value):
        return True
    return not str(value).strip()


def _integer(value: Any, field: str, reference_id: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CliffReviewValidationError(
            f"{reference_id}: {field} must be an integer"
        ) from exc
    if not isfinite(number) or not number.is_integer():
        raise CliffReviewValidationError(
            f"{reference_id}: {field} must be an integer"
        )
    return int(number)


def validate_review_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_stints: Mapping[str, Mapping[str, Any]] | None = None,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Validate manual labels and return normalized review rows."""
    normalized = []
    seen = set()
    for index, row in enumerate(rows):
        reference_id = str(row.get("reference_id") or "").strip()
        if not reference_id:
            raise CliffReviewValidationError(
                f"review row {index + 1} requires reference_id"
            )
        if reference_id in seen:
            raise CliffReviewValidationError(
                f"duplicate reviewed reference_id: {reference_id}"
            )
        seen.add(reference_id)
        if source_stints is not None and reference_id not in source_stints:
            raise CliffReviewValidationError(
                f"{reference_id}: not present in source references"
            )
        if str(row.get("split") or "") != "calibration":
            raise CliffReviewValidationError(
                f"{reference_id}: only calibration labels may be reviewed"
            )

        status = str(row.get("manual_review_status") or "").strip()
        if status not in REVIEW_STATUSES:
            raise CliffReviewValidationError(
                f"{reference_id}: unsupported manual_review_status {status!r}"
            )
        cliff_value = row.get("reviewed_cliff_lap")
        notes = row.get("review_notes")
        if status == "pending" and require_complete:
            raise CliffReviewValidationError(
                f"{reference_id}: review is still pending"
            )
        if status == "confirmed_cliff":
            cliff_lap = _integer(
                cliff_value,
                "reviewed_cliff_lap",
                reference_id,
            )
            start = _integer(
                row.get("starting_tyre_age"),
                "starting_tyre_age",
                reference_id,
            )
            end = _integer(
                row.get("ending_tyre_age"),
                "ending_tyre_age",
                reference_id,
            )
            if not start <= cliff_lap <= end:
                raise CliffReviewValidationError(
                    f"{reference_id}: reviewed cliff lap must be within "
                    f"{start}–{end}"
                )
        else:
            if not _blank(cliff_value):
                raise CliffReviewValidationError(
                    f"{reference_id}: {status} must not define reviewed_cliff_lap"
                )
            cliff_lap = None
        if status == "rejected" and _blank(notes):
            raise CliffReviewValidationError(
                f"{reference_id}: rejected reviews require review_notes"
            )

        normalized.append(
            {
                **dict(row),
                "reference_id": reference_id,
                "manual_review_status": status,
                "reviewed_cliff_lap": cliff_lap,
                "review_notes": None if _blank(notes) else str(notes).strip(),
            }
        )
    if not normalized:
        raise CliffReviewValidationError("review file contains no rows")
    return normalized


def apply_review_decisions(
    queue_rows: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a partial set of manual decisions to an existing review queue."""
    queue = [dict(row) for row in queue_rows]
    by_id = {
        str(row.get("reference_id") or "").strip(): row
        for row in queue
    }
    if len(by_id) != len(queue) or "" in by_id:
        raise CliffReviewValidationError(
            "review queue reference_id values must be present and unique"
        )

    seen = set()
    for decision in decisions:
        reference_id = str(decision.get("reference_id") or "").strip()
        if reference_id in seen:
            raise CliffReviewValidationError(
                f"duplicate decision reference_id: {reference_id}"
            )
        seen.add(reference_id)
        if reference_id not in by_id:
            raise CliffReviewValidationError(
                f"{reference_id}: decision is not present in the review queue"
            )
        row = by_id[reference_id]
        row["manual_review_status"] = decision.get("manual_review_status")
        row["reviewed_cliff_lap"] = decision.get("reviewed_cliff_lap")
        row["review_notes"] = decision.get("review_notes")

    validate_review_rows(queue, require_complete=False)
    return queue
