"""Persistent storage for processed FastF1 training sessions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd


SCHEMA_VERSION = 1
ACTIVE_AERO_ROLE = "active_aero"
PHYSICS_PRIOR_ROLE = "physics_prior"
GROUND_EFFECT_ROLE = "ground_effect"
SESSION_WEIGHTS = {
    "R": 1.00,
    "FP2": 0.50,
    "S": 0.75,
}


class TrainingDataStoreError(RuntimeError):
    """Raised when processed training data is missing or invalid."""


@dataclass(frozen=True)
class SessionSpec:
    year: int
    round_number: int
    event_name: str
    event_date: str
    session_code: str
    role: str

    @property
    def key(self) -> str:
        return f"{self.year}:{self.event_name}:{self.session_code}"

    @property
    def weight(self) -> float:
        return SESSION_WEIGHTS[self.session_code]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_fingerprint(columns: Iterable[str]) -> str:
    encoded = "\n".join(str(column) for column in columns).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return cleaned or "event"


def _as_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _event_date(row: pd.Series) -> Optional[date]:
    for column in ("EventDate", "Session5Date", "Session5DateUtc"):
        if column in row:
            parsed = _as_date(row[column])
            if parsed is not None:
                return parsed
    return None


def _available_session_codes(row: pd.Series) -> List[str]:
    codes = ["R"]
    session_names = [
        str(row.get(f"Session{index}", "")).strip().lower()
        for index in range(1, 6)
    ]
    if any(name == "practice 2" for name in session_names):
        codes.append("FP2")
    if any(name in {"sprint", "sprint race"} for name in session_names):
        codes.append("S")
    return codes


def session_specs_from_schedule(
    schedule: pd.DataFrame,
    *,
    year: int,
    as_of: date,
    role: str,
) -> List[SessionSpec]:
    """Convert a FastF1 event schedule into completed, relevant session specs."""
    required_columns = {"RoundNumber", "EventName"}
    if schedule is None or schedule.empty or not required_columns.issubset(schedule.columns):
        raise TrainingDataStoreError(
            f"FastF1 schedule for {year} is empty or missing RoundNumber/EventName"
        )

    specs: List[SessionSpec] = []
    for _, row in schedule.iterrows():
        try:
            round_number = int(row.get("RoundNumber"))
        except (TypeError, ValueError):
            continue
        event_name = row.get("EventName")
        completed_on = _event_date(row)
        if (
            round_number <= 0
            or event_name is None
            or pd.isna(event_name)
            or completed_on is None
            or completed_on > as_of
        ):
            continue

        for session_code in _available_session_codes(row):
            specs.append(
                SessionSpec(
                    year=year,
                    round_number=round_number,
                    event_name=str(event_name),
                    event_date=completed_on.isoformat(),
                    session_code=session_code,
                    role=role,
                )
            )

    return sorted(
        specs,
        key=lambda item: (
            item.year,
            item.round_number,
            0 if item.session_code == "R" else 1,
            item.session_code,
        ),
    )


class ProcessedSessionStore:
    """Store each processed session as an immutable compressed CSV plus a manifest."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self._manifest = self._read_manifest()

    def _empty_manifest(self) -> Dict[str, Any]:
        now = _utc_timestamp()
        return {
            "schema_version": SCHEMA_VERSION,
            "format": "csv.gz",
            "description": (
                "Processed FastF1 sessions used for reproducible model training. "
                "Raw FastF1 response files are stored separately."
            ),
            "created_at": now,
            "updated_at": now,
            "sessions": {},
        }

    def _read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingDataStoreError(
                f"Could not read processed-data manifest {self.manifest_path}: {exc}"
            ) from exc
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise TrainingDataStoreError(
                "Unsupported processed-data manifest schema: "
                f"{payload.get('schema_version')!r}"
            )
        if not isinstance(payload.get("sessions"), dict):
            raise TrainingDataStoreError("Processed-data manifest sessions must be an object")
        return payload

    @property
    def sessions(self) -> Dict[str, Dict[str, Any]]:
        return self._manifest["sessions"]

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._manifest.get(key, default)

    def manifest_sha256(self) -> str:
        if not self.manifest_path.is_file():
            raise TrainingDataStoreError(
                f"Processed-data manifest is missing: {self.manifest_path}"
            )
        return _sha256(self.manifest_path)

    def has_session(self, key: str) -> bool:
        return key in self.sessions

    def session_keys(self, role: Optional[str] = None) -> List[str]:
        return sorted(
            key
            for key, entry in self.sessions.items()
            if role is None or entry.get("role") == role
        )

    def _relative_path(self, spec: SessionSpec) -> Path:
        key_hash = hashlib.sha256(spec.key.encode("utf-8")).hexdigest()[:10]
        filename = (
            f"{spec.year}-{spec.round_number:02d}-{_slug(spec.event_name)}-"
            f"{spec.session_code.lower()}-{key_hash}.csv.gz"
        )
        return Path(spec.role) / str(spec.year) / filename

    def _write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest["updated_at"] = _utc_timestamp()
        ordered_sessions = {
            key: self.sessions[key]
            for key in sorted(self.sessions)
        }
        payload = dict(self._manifest)
        payload["sessions"] = ordered_sessions
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)
        self._manifest = payload

    def set_metadata(self, key: str, value: Any) -> bool:
        if key in {"schema_version", "format", "sessions"}:
            raise TrainingDataStoreError(f"Manifest key {key!r} is reserved")
        if self._manifest.get(key) == value:
            return False
        self._manifest[key] = value
        self._write_manifest()
        return True

    def save_session(self, spec: SessionSpec, frame: pd.DataFrame) -> bool:
        """Save a new processed session. Existing session keys are immutable."""
        if self.has_session(spec.key):
            return False
        if frame is None or frame.empty:
            raise TrainingDataStoreError(f"Cannot save empty processed session {spec.key}")

        output = frame.copy()
        output["SampleWeight"] = spec.weight
        if "EventDate" in output.columns:
            output["EventDate"] = pd.to_datetime(output["EventDate"]).dt.strftime("%Y-%m-%d")

        relative_path = self._relative_path(spec)
        output_path = self.root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        output.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        os.replace(temporary, output_path)

        self.sessions[spec.key] = {
            **asdict(spec),
            "session_key": spec.key,
            "session_weight": spec.weight,
            "path": relative_path.as_posix(),
            "rows": int(len(output)),
            "columns": int(len(output.columns)),
            "columns_sha256": _column_fingerprint(output.columns),
            "sha256": _sha256(output_path),
            "processed_at": _utc_timestamp(),
        }
        self._write_manifest()
        return True

    def validate(self, *, verify_hashes: bool = False) -> None:
        for key, entry in self.sessions.items():
            relative_path = entry.get("path")
            if not relative_path:
                raise TrainingDataStoreError(f"Session {key} has no data path")
            path = self.root / relative_path
            if not path.is_file():
                raise TrainingDataStoreError(f"Session data file is missing: {path}")
            if verify_hashes and _sha256(path) != entry.get("sha256"):
                raise TrainingDataStoreError(f"Session data checksum mismatch: {path}")

    def load_role(self, role: str) -> pd.DataFrame:
        entries = [
            self.sessions[key]
            for key in self.session_keys(role)
        ]
        if not entries:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for entry in entries:
            path = self.root / entry["path"]
            if not path.is_file():
                raise TrainingDataStoreError(f"Session data file is missing: {path}")
            frame = pd.read_csv(path, compression="gzip")
            if len(frame) != int(entry["rows"]):
                raise TrainingDataStoreError(
                    f"Session row count does not match manifest for {entry['session_key']}"
                )
            # Preprocessing one-hot encodes EventName. Restore immutable event
            # metadata from the manifest so chronological validation can keep
            # repeated annual events in distinct walk-forward folds.
            if "EventName" not in frame.columns:
                frame["EventName"] = entry["event_name"]
            if "EventDate" not in frame.columns:
                frame["EventDate"] = entry["event_date"]
            frame["SessionKey"] = entry["session_key"]
            frame["SessionCode"] = entry["session_code"]
            frame["TrainingRole"] = entry["role"]
            frame["Season"] = int(entry["year"])
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        if "EventDate" in combined.columns:
            combined["EventDate"] = pd.to_datetime(
                combined["EventDate"], errors="coerce"
            )
        return combined.fillna(0)


def refresh_sessions(
    store: ProcessedSessionStore,
    specs: Iterable[SessionSpec],
    *,
    loader: Callable[[int, str, str], Any],
    preprocessor: Callable[[Any], pd.DataFrame],
) -> Dict[str, Any]:
    """Download/process only sessions that are not already present in the store."""
    added: List[str] = []
    skipped: List[str] = []
    failures: List[Dict[str, str]] = []

    for spec in specs:
        if store.has_session(spec.key):
            skipped.append(spec.key)
            continue
        try:
            session = loader(spec.year, spec.event_name, spec.session_code)
            frame = preprocessor(session)
            store.save_session(spec, frame)
            added.append(spec.key)
        except Exception as exc:
            failures.append(
                {
                    "session_key": spec.key,
                    "session_code": spec.session_code,
                    "error": str(exc),
                }
            )

    mandatory_failures = [
        failure for failure in failures if failure["session_code"] == "R"
    ]
    return {
        "added": added,
        "skipped": skipped,
        "failures": failures,
        "mandatory_failures": mandatory_failures,
    }
