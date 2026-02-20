import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import fastf1
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data_loader import load_race_data
from preprocessing import preprocess_laps


def _to_event_date(value) -> date:
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return date.max


def _list_completed_events(year: int, as_of: date) -> List[str]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if "RoundNumber" not in schedule.columns:
        return []
    races = schedule[schedule["RoundNumber"].notna()].copy()
    if "EventDate" in races.columns:
        races = races[races["EventDate"].apply(_to_event_date) <= as_of]
    races = races.sort_values("RoundNumber")
    return [str(v) for v in races["EventName"].dropna().tolist()]


def collect_era_data(start_year: int, end_year: int, as_of: date) -> Tuple[pd.DataFrame, Dict]:
    frames: List[pd.DataFrame] = []
    loaded_events: List[str] = []
    failed_events: List[str] = []

    for year in range(start_year, end_year + 1):
        try:
            events = _list_completed_events(year, as_of)
        except Exception as exc:
            failed_events.append(f"{year}:schedule:{exc}")
            continue

        for event_name in events:
            key = f"{year}:{event_name}"
            try:
                session = load_race_data(year, event_name, "R")
                df = preprocess_laps(session)
                if df.empty:
                    failed_events.append(f"{key}:empty")
                    continue
                frames.append(df)
                loaded_events.append(key)
                print(f"Loaded {key} -> {len(df)} processed laps")
            except Exception as exc:
                failed_events.append(f"{key}:{exc}")
                print(f"Failed {key}: {exc}")

    if not frames:
        return pd.DataFrame(), {"loaded_events": loaded_events, "failed_events": failed_events}
    return pd.concat(frames, ignore_index=True), {"loaded_events": loaded_events, "failed_events": failed_events}


def train_and_save(data_df: pd.DataFrame, model_path: Path, features_path: Path) -> Dict:
    X = data_df.drop("LapTimeSeconds", axis=1)
    y = data_df["LapTimeSeconds"]

    model = HistGradientBoostingRegressor(random_state=42)
    if len(data_df) >= 50:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    else:
        model.fit(X, y)
        rmse = None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), features_path)

    return {
        "rows": int(len(data_df)),
        "features": int(X.shape[1]),
        "rmse": rmse,
    }


def train_era(
    start_year: int,
    end_year: int,
    output_prefix: str,
    as_of: date,
    output_dir: Path,
) -> Dict:
    print(f"Collecting data for {start_year}-{end_year} as of {as_of.isoformat()}...")
    data_df, details = collect_era_data(start_year, end_year, as_of)
    if data_df.empty:
        return {
            "status": "no_data",
            "start_year": start_year,
            "end_year": end_year,
            "as_of": as_of.isoformat(),
            **details,
        }

    model_path = output_dir / f"{output_prefix}_model.joblib"
    features_path = output_dir / f"{output_prefix}_features.joblib"
    metrics = train_and_save(data_df, model_path, features_path)
    return {
        "status": "trained",
        "start_year": start_year,
        "end_year": end_year,
        "as_of": as_of.isoformat(),
        "model_path": str(model_path),
        "features_path": str(features_path),
        **metrics,
        **details,
    }


def main():
    parser = argparse.ArgumentParser(description="Train era-specific F1 tire degradation models")
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=date.today().isoformat(),
        help="Only races completed on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models and metadata",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "ground_effect", "active_aero"],
        default="both",
        help="Which era model(s) to train",
    )
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of_date)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    if args.mode in {"both", "ground_effect"}:
        results["ground_effect_2022_2025"] = train_era(
            2022, 2025, "ground_effect_2022_2025", as_of, out_dir
        )
    if args.mode in {"both", "active_aero"}:
        results["active_aero_2026_2030"] = train_era(
            2026, 2030, "active_aero_2026_2030", as_of, out_dir
        )

    metadata_path = out_dir / "era_training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
