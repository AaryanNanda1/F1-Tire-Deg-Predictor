import unittest

from cliff_review import (
    CliffReviewValidationError,
    apply_review_decisions,
    build_review_queue,
    validate_review_rows,
)


def _stint(
    reference_id,
    *,
    race_id="calibration-race-a",
    compound="MEDIUM",
    status="candidate_for_manual_review",
    confidence="medium",
):
    return {
        "reference_id": reference_id,
        "race_id": race_id,
        "split": "calibration",
        "season": 2023,
        "event_name": "Test Grand Prix",
        "driver": "TST",
        "stint": 1,
        "compound": compound,
        "starting_tyre_age": 1,
        "ending_tyre_age": 20,
        "clean_lap_count": 18,
        "removed_fraction": 0.1,
        "reference_status": status,
        "reference_confidence": confidence,
        "observed_cliff_lap": 12 if status != "no_cliff_candidate" else None,
    }


class CliffReviewTest(unittest.TestCase):
    def test_queue_includes_all_high_and_balances_samples(self):
        references = {
            "schema_version": 1,
            "accepted_stints": [
                _stint("high", confidence="high"),
                _stint("medium-soft", compound="SOFT"),
                _stint(
                    "medium-hard",
                    race_id="calibration-race-b",
                    compound="HARD",
                ),
                _stint(
                    "no-cliff",
                    status="no_cliff_candidate",
                    confidence=None,
                ),
            ],
        }

        queue = build_review_queue(
            references,
            medium_limit=1,
            no_cliff_limit=1,
        )

        self.assertEqual(queue[0]["reference_id"], "high")
        self.assertEqual(len(queue), 3)
        self.assertTrue(
            all(row["manual_review_status"] == "pending" for row in queue)
        )

    def test_queue_rejects_holdout_stints(self):
        holdout = _stint("holdout")
        holdout["split"] = "holdout"
        with self.assertRaisesRegex(
            CliffReviewValidationError,
            "calibration stints only",
        ):
            build_review_queue(
                {"schema_version": 1, "accepted_stints": [holdout]}
            )

    def test_validates_complete_labels(self):
        rows = [
            {
                **_stint("cliff"),
                "manual_review_status": "confirmed_cliff",
                "reviewed_cliff_lap": 13,
                "review_notes": "sustained pace break",
            },
            {
                **_stint("no-cliff"),
                "manual_review_status": "confirmed_no_cliff",
                "reviewed_cliff_lap": None,
                "review_notes": "",
            },
        ]

        normalized = validate_review_rows(rows)

        self.assertEqual(normalized[0]["reviewed_cliff_lap"], 13)
        self.assertIsNone(normalized[1]["reviewed_cliff_lap"])

    def test_rejects_cliff_outside_observed_range(self):
        row = {
            **_stint("bad-cliff"),
            "manual_review_status": "confirmed_cliff",
            "reviewed_cliff_lap": 25,
            "review_notes": "",
        }
        with self.assertRaisesRegex(
            CliffReviewValidationError,
            "within 1–20",
        ):
            validate_review_rows([row])

    def test_requires_notes_for_rejected_review(self):
        row = {
            **_stint("rejected"),
            "manual_review_status": "rejected",
            "reviewed_cliff_lap": None,
            "review_notes": "",
        }
        with self.assertRaisesRegex(
            CliffReviewValidationError,
            "require review_notes",
        ):
            validate_review_rows([row])

    def test_applies_partial_decisions_and_preserves_pending_rows(self):
        queue = [
            {
                **_stint("reviewed"),
                "manual_review_status": "pending",
                "reviewed_cliff_lap": None,
                "review_notes": None,
            },
            {
                **_stint("pending"),
                "manual_review_status": "pending",
                "reviewed_cliff_lap": None,
                "review_notes": None,
            },
        ]
        decisions = [
            {
                "reference_id": "reviewed",
                "manual_review_status": "confirmed_cliff",
                "reviewed_cliff_lap": 12,
                "review_notes": "visible sustained break",
            }
        ]

        updated = apply_review_decisions(queue, decisions)

        self.assertEqual(updated[0]["manual_review_status"], "confirmed_cliff")
        self.assertEqual(updated[0]["reviewed_cliff_lap"], 12)
        self.assertEqual(updated[1]["manual_review_status"], "pending")


if __name__ == "__main__":
    unittest.main()
