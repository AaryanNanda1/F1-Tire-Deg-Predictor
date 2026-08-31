#!/usr/bin/env python3
"""Reprocess every session in a persistent store with the current schema.

The command writes to a new output directory. It never mutates the source
store, model artifacts, or deployment files, which makes a large FastF1
migration restartable and safe to validate before cutover.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import fastf1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import load_race_data
from preprocessing import preprocess_laps
from training_data_store import (
    ProcessedSessionStore,
    SessionSpec,
    TrainingDataStoreError,
    refresh_sessions,
)


def _write_json(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _specs_from_manifest(source: ProcessedSessionStore) -> List[SessionSpec]:
    specs: List[SessionSpec] = []
    for key in source.session_keys():
        entry = source.sessions[key]
        specs.append(
            SessionSpec(
                year=int(entry["year"]),
                round_number=int(entry["round_number"]),
                event_name=str(entry["event_name"]),
                event_date=str(entry["event_date"]),
                session_code=str(entry["session_code"]),
                role=str(entry["role"]),
            )
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reprocess all sessions from an existing persistent FastF1 store "
            "using the current preprocessing and manifest schema."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--offline-cache-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing partially rebuilt output directory",
    )
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_sessions < 0:
        print("--max-sessions cannot be negative", file=sys.stderr)
        return 2

    source = ProcessedSessionStore(args.source_dir)
    output_path = Path(args.output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not args.resume:
        raise SystemExit(f"Output directory must be empty or absent: {output_path}")
    if args.offline_cache_only:
        fastf1.Cache.offline_mode(True)

    specs = _specs_from_manifest(source)
    if args.max_sessions:
        specs = specs[: args.max_sessions]
    target = ProcessedSessionStore(output_path)
    if args.resume and target.requires_rebuild:
        raise SystemExit(
            f"Cannot resume a directory with a stale/incomplete manifest: {output_path}"
        )
    # Preserve store-level coverage metadata while replacing the rows and
    # upgrading the schema/provenance fields in the new manifest.
    for metadata_key in ("coverage",):
        metadata_value = source.get_metadata(metadata_key)
        if metadata_value is not None:
            target.set_metadata(metadata_key, metadata_value)
    result = refresh_sessions(
        target,
        specs,
        loader=load_race_data,
        preprocessor=preprocess_laps,
    )
    target.validate(verify_hashes=True)
    payload = {
        "status": "ok" if not result["mandatory_failures"] else "failed",
        "source_dir": str(args.source_dir),
        "output_dir": str(output_path),
        "source_schema_version": source.manifest_schema_version,
        "source_requires_rebuild": source.requires_rebuild,
        "requested_sessions": len(specs),
        "target_schema_version": target.manifest_schema_version,
        "target_preprocessing_version": target.get_metadata("preprocessing_version"),
        "target_preprocessing_git_commit": target.get_metadata("preprocessing_git_commit"),
        "update": result,
    }
    _write_json(args.json_output, payload)
    print(json.dumps(payload, indent=2))
    return 2 if payload["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
