import argparse
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import fastf1
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data_loader import load_race_data
from preprocessing import preprocess_laps
from training_data_store import (
    ACTIVE_AERO_ROLE,
    GROUND_EFFECT_ROLE,
    PHYSICS_PRIOR_ROLE,
    ProcessedSessionStore,
    TrainingDataStoreError,
)


TRAINING_METADATA_COLUMNS = [
    "LapTimeDelta",
    "SampleWeight",
    "EventDate",
    "EventName",
    "SessionKey",
    "SessionCode",
    "TrainingRole",
    "Season",
]


def _training_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _merge_training_result(existing: Dict, candidate: Dict) -> Dict:
    """Keep the last usable model metadata when a retraining attempt has no data."""
    candidate_status = candidate.get("status", "")
    existing_status = existing.get("status", "")
    if candidate_status == "no_data" and existing_status.startswith("trained"):
        preserved = dict(existing)
        preserved["last_retrain_attempt"] = candidate.get("as_of")
        preserved["last_retrain_status"] = candidate_status
        preserved["last_retrain_failed_events"] = candidate.get("failed_events", [])
        return preserved
    return candidate


def _to_event_date(value) -> date:
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return date.max


def _list_completed_events(year: int, as_of: date) -> List[str]:
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        return []
        
    if schedule is None or schedule.empty or "RoundNumber" not in schedule.columns:
        return []
        
    races = schedule[schedule["RoundNumber"].notna()].copy()
    if "EventDate" in races.columns:
        # Filter for completed events specifically
        # Ensure we drop rows with invalid dates before comparison
        races = races[races["EventDate"].notna()]
        races = races[races["EventDate"].dt.normalize().dt.date <= as_of]
        
    races = races.sort_values("RoundNumber")
    # We only care about Grand Prix events (ignore testing/optional sessions if they slipped through)
    return [str(v) for v in races["EventName"].dropna().tolist()]


def collect_era_data(start_year: int, end_year: int, as_of: date) -> Tuple[pd.DataFrame, Dict]:
    frames: List[pd.DataFrame] = []
    loaded_events: List[str] = []
    failed_events: List[str] = []

    # Session weights: Race=1.0 (primary), FP2=0.5, Sprint=0.75, Qualifying=EXCLUDED
    SESSION_CONFIG = [
        ("R",   1.00),   # Race: primary training source
        ("FP2", 0.50),   # Practice Day 2: long-run simulations, most race-representative practice
        ("S",   0.75),   # Sprint: race conditions, lower fuel than GP but more representative than FP
        # FP1/FP3 excluded: typically short runs, setup testing, less useful for tire deg
        # Q excluded: low fuel, push laps, completely different to race tire management
    ]

    for year in range(start_year, end_year + 1):
        try:
            events = _list_completed_events(year, as_of)
        except Exception as exc:
            failed_events.append(f"{year}:schedule:{exc}")
            continue

        for event_name in events:
            for session_code, weight in SESSION_CONFIG:
                key = f"{year}:{event_name}:{session_code}"
                try:
                    session = load_race_data(year, event_name, session_code)
                    df = preprocess_laps(session)
                    if df.empty:
                        failed_events.append(f"{key}:empty")
                        continue
                    df['SampleWeight'] = weight
                    frames.append(df)
                    loaded_events.append(key)
                    print(f"Loaded {key} (weight={weight}) -> {len(df)} processed laps")
                except Exception as exc:
                    # Non-race sessions are optional; silently skip if unavailable
                    if session_code == "R":
                        failed_events.append(f"{key}:{exc}")
                        print(f"Failed {key}: {exc}")
                    # else: FP2/Sprint missing is expected for some events, no need to log

    if not frames:
        return pd.DataFrame(), {"loaded_events": loaded_events, "failed_events": failed_events}
    
    # Combine frames
    final_df = pd.concat(frames, ignore_index=True)
    
    # --- Soft-Tire Specific Age Weighting ---
    # Exponentially increase the weight of soft tires as they age to force the model
    # to pay attention to the degradation "cliff" (Recommendation 2)
    final_df['CompoundWeight'] = 1.0
    # preprocessing.py uses pd.get_dummies, so 'Compound' is removed and 'Compound_SOFT' is created
    if 'Compound_SOFT' in final_df.columns:
        soft_mask = (final_df['Compound_SOFT'] == 1)
        # Formula: 1.0 base + 0.1 per lap of age. A 20-lap old soft gets 3.0x weight.
        final_df.loc[soft_mask, 'CompoundWeight'] = 1.0 + (final_df.loc[soft_mask, 'TyreLife'] / 10.0)
        
    final_df['SampleWeight'] = final_df['SampleWeight'] * final_df['CompoundWeight']
    # Clean up the temporary column
    final_df.drop(columns=['CompoundWeight'], inplace=True)
    
    return final_df, {"loaded_events": loaded_events, "failed_events": failed_events}


def _apply_soft_tire_age_weighting(data_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the original production soft-tire weighting to processed rows."""
    weighted = data_df.copy()
    if "SampleWeight" not in weighted.columns:
        weighted["SampleWeight"] = 1.0
    if "Compound_SOFT" in weighted.columns and "TyreLife" in weighted.columns:
        soft_mask = weighted["Compound_SOFT"] == 1
        weighted.loc[soft_mask, "SampleWeight"] *= (
            1.0 + weighted.loc[soft_mask, "TyreLife"] / 10.0
        )
    return weighted


def load_persistent_active_aero_data(
    processed_data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load repository-versioned Active Aero and frozen prior sessions."""
    store = ProcessedSessionStore(processed_data_dir)
    store.validate(verify_hashes=True)
    active_data = store.load_role(ACTIVE_AERO_ROLE)
    prior_data = store.load_role(PHYSICS_PRIOR_ROLE)

    if active_data.empty:
        raise TrainingDataStoreError(
            f"No Active Aero sessions are stored in {processed_data_dir}"
        )
    if prior_data.empty:
        raise TrainingDataStoreError(
            f"No physics-prior sessions are stored in {processed_data_dir}"
        )

    return active_data, prior_data, {
        "loaded_events": store.session_keys(ACTIVE_AERO_ROLE),
        "failed_events": [],
        "training_data_source": "persistent_processed_sessions",
        "processed_data_manifest": str(store.manifest_path),
        "processed_data_manifest_sha256": store.manifest_sha256(),
        "active_session_count": len(store.session_keys(ACTIVE_AERO_ROLE)),
        "prior_session_count": len(store.session_keys(PHYSICS_PRIOR_ROLE)),
    }


def load_persistent_ground_effect_data(
    processed_data_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    """Load the complete repository-versioned 2022-2025 dataset."""
    store = ProcessedSessionStore(processed_data_dir)
    store.validate(verify_hashes=True)
    coverage = store.get_metadata("coverage", {})
    if not coverage.get("complete"):
        raise TrainingDataStoreError(
            "Ground Effect dataset coverage is incomplete: "
            f"{coverage.get('missing_race_count', 'unknown')} races and "
            f"{coverage.get('missing_session_count', 'unknown')} sessions remain"
        )
    data = store.load_role(GROUND_EFFECT_ROLE)
    if data.empty:
        raise TrainingDataStoreError(
            f"No Ground Effect sessions are stored in {processed_data_dir}"
        )

    return _apply_soft_tire_age_weighting(data), {
        "loaded_events": store.session_keys(GROUND_EFFECT_ROLE),
        "failed_events": [],
        "training_data_source": "persistent_processed_sessions",
        "processed_data_manifest": str(store.manifest_path),
        "processed_data_manifest_sha256": store.manifest_sha256(),
        "session_count": len(store.session_keys(GROUND_EFFECT_ROLE)),
        "race_count": int(coverage.get("expected_race_count", 0)),
        "coverage_start_year": 2022,
        "coverage_end_year": 2025,
    }


def build_active_aero_training_data(
    active_data: pd.DataFrame,
    prior_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Reproduce the original single-model Active Aero blending policy."""
    if active_data.empty:
        return pd.DataFrame(), 0

    active_weighted = _apply_soft_tire_age_weighting(active_data)
    frames = [active_weighted]
    sampled_prior_rows = 0
    if not prior_data.empty:
        sampled_prior = prior_data.sample(frac=0.50, random_state=42).copy()
        sampled_prior["SampleWeight"] = 1.0
        sampled_prior_rows = len(sampled_prior)
        frames.append(sampled_prior)

    data_df = pd.concat(frames, ignore_index=True, sort=False).fillna(0)
    if "IsWet" in data_df.columns:
        wet_laps = data_df[data_df["IsWet"] == 1]
        if not wet_laps.empty and len(wet_laps) < 500:
            print(f"  Upsampling wet/inter conditions (found {len(wet_laps)} laps)...")
            data_df = pd.concat([data_df, wet_laps, wet_laps], ignore_index=True)

    return data_df, sampled_prior_rows


def train_and_save(
    data_df: pd.DataFrame,
    model_path: Path,
    features_path: Path,
    prior_model_path: Path = None,
    prior_weight: float = 0.20,
    validation_test_start_year: int = None,
) -> Dict:
    """
    Train and save a tire degradation model.

    Cold-start prior: if fewer than MIN_RACES_FOR_FULL_TRAINING distinct races exist
    in data_df, synthetic rows are generated by running the prior (Ground Effect) model
    on the same input space and injecting them at a low sample weight. This prevents
    the new-era model from being wildly underfit while it accumulates race data.
    """
    MIN_RACES_FOR_FULL_TRAINING = 5  # Below this, activate cold-start prior
    
    # Count distinct race events to decide if cold start is needed
    race_laps = data_df[data_df.get('SampleWeight', pd.Series(1.0, index=data_df.index)) == 1.0] \
        if 'SampleWeight' in data_df.columns else data_df
    n_race_events = data_df.shape[0] // 300 if len(data_df) > 0 else 0  # rough estimate
    
    cold_start_active = False
    if prior_model_path and prior_model_path.exists() and n_race_events < MIN_RACES_FOR_FULL_TRAINING:
        print(f"  Cold-start prior active: only ~{n_race_events} races available (< {MIN_RACES_FOR_FULL_TRAINING}).")
        print(f"  Blending Ground Effect prior at {int(prior_weight*100)}% weight to stabilize predictions.")
        try:
            prior_model = joblib.load(prior_model_path)
            prior_features = joblib.load(str(prior_model_path).replace('_model.joblib', '_features.joblib'))
            
            # Build prior synthetic rows from the current era data inputs
            X_current = data_df.drop(
                columns=TRAINING_METADATA_COLUMNS,
                errors="ignore",
            )
            X_prior_aligned = X_current.reindex(columns=prior_features, fill_value=0)
            prior_preds = prior_model.predict(X_prior_aligned)
            
            prior_df = data_df.copy()
            prior_df['LapTimeSeconds'] = prior_preds
            prior_df['SampleWeight'] = prior_weight
            
            data_df = pd.concat([data_df, prior_df], ignore_index=True)
            cold_start_active = True
            print(f"  Injected {len(prior_df)} synthetic prior rows.")
        except Exception as e:
            print(f"  Warning: could not load prior model, skipping cold start: {e}")

    # 1. Chronological Validation (Walk-Forward)
    # This evaluates how the model would have performed "in-season"
    sequential_mae = None
    if len(data_df) > 500: # Only worth doing if we have enough data
        try:
            sequential_mae = perform_walk_forward_validation(
                data_df,
                test_start_year=validation_test_start_year,
            )
        except Exception as e:
            print(f"  Warning: Sequential validation failed: {e}")

    # Extract sample weights for final training
    sample_weights = data_df.get('SampleWeight', pd.Series(1.0, index=data_df.index))
    
    # Drop target and metadata columns so they aren't used as features
    X = data_df.drop(columns=TRAINING_METADATA_COLUMNS, errors="ignore")
    y = data_df['LapTimeDelta']

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    
    # Fit the FINAL production model on ALL data
    model.fit(X, y, sample_weight=sample_weights)
    
    # Use the sequential_mae if available, otherwise fallback to simple training MAE
    final_mae = sequential_mae if sequential_mae is not None else float(mean_absolute_error(y, model.predict(X)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), features_path)

    return {
        "rows": int(len(data_df)),
        "features": int(X.shape[1]),
        "mae": round(float(final_mae), 3),
        "mae_validation_scope": (
            f"walk_forward_{validation_test_start_year}_plus_test_events"
            if sequential_mae is not None
            and validation_test_start_year is not None
            else (
                "walk_forward_all_test_events"
                if sequential_mae is not None
                else "in_sample_fallback"
            )
        ),
        "cold_start_prior_active": cold_start_active,
    }


def perform_walk_forward_validation(
    df: pd.DataFrame,
    *,
    test_start_year: int = None,
) -> float:
    """
    Simulates "Next Race" prediction by iteratively training on past events 
    and testing on the very next one.
    """
    # Check if we have the necessary metadata
    if 'EventDate' not in df.columns or 'EventName' not in df.columns:
        return 0.0
        
    # Group by EventDate and EventName to get unique race weekends in order
    events = df[['EventDate', 'EventName']].drop_duplicates().copy()
    events['EventDate'] = pd.to_datetime(events['EventDate'], errors='coerce')
    events = events.dropna(subset=['EventDate']).sort_values(['EventDate', 'EventName'])
    if test_start_year is not None:
        scored_events = events[events['EventDate'].dt.year >= test_start_year]
    else:
        scored_events = events.iloc[2:]
    
    if len(events) < 2:
        return 0.0
        
    print(
        "\n  [Walk-Forward Validation] "
        f"Scoring {len(scored_events)} of {len(events)} chronological events..."
    )
    
    maes = []
    event_dates = pd.to_datetime(df['EventDate'], errors='coerce')
    for _, test_event in scored_events.iterrows():
        train_data = df[event_dates < test_event['EventDate']]
        test_data = df[
            (event_dates == test_event['EventDate'])
            & (df['EventName'] == test_event['EventName'])
        ]
        if train_data.empty or test_data.empty:
            continue
        
        # Ensure we drop metadata columns during training
        X_train = train_data.drop(
            columns=TRAINING_METADATA_COLUMNS,
            errors="ignore",
        )
        y_train = train_data['LapTimeDelta']
        w_train = train_data.get('SampleWeight', pd.Series(1.0, index=train_data.index))
        
        X_test = test_data.drop(
            columns=TRAINING_METADATA_COLUMNS,
            errors="ignore",
        )
        y_test = test_data['LapTimeDelta']
        
        # We don't use early stopping here to speed up validation loops
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=50, random_state=42)
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        maes.append(mae)
        print(f"    Tested on {test_event['EventName']} ({test_event['EventDate']}) -> MAE: {mae:.3f}s")

    overall_mae = np.mean(maes) if maes else 0.0
    print(f"  [Walk-Forward Validation] Overall Sequential MAE: {overall_mae:.3f}s\n")
    return overall_mae


def train_era(
    start_year: int,
    end_year: int,
    output_prefix: str,
    as_of: date,
    output_dir: Path,
    prior_model_path: Path = None,
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
    metrics = train_and_save(data_df, model_path, features_path, prior_model_path=prior_model_path)
    
    status = "trained_cold_start" if metrics.get("cold_start_prior_active") else "trained"
    return {
        "status": status,
        "start_year": start_year,
        "end_year": end_year,
        "as_of": as_of.isoformat(),
        "trained_at": _training_timestamp(),
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
    parser.add_argument(
        "--processed-data-dir",
        type=str,
        default=None,
        help=(
            "Train the selected single era from a persistent processed-session "
            "store instead of reloading historical FastF1 sessions"
        ),
    )
    args = parser.parse_args()

    if args.processed_data_dir and args.mode == "both":
        parser.error(
            "--processed-data-dir requires --mode ground_effect or --mode active_aero"
        )

    as_of = date.fromisoformat(args.as_of_date)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "era_training_metadata.json"
    existing_results: Dict[str, Dict] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except (OSError, json.JSONDecodeError):
            existing_results = {}

    # Preserve metadata for eras that are not part of this invocation.
    results: Dict[str, Dict] = dict(existing_results)
    if args.mode in {"both", "ground_effect"}:
        if args.processed_data_dir:
            data_df, details = load_persistent_ground_effect_data(
                Path(args.processed_data_dir)
            )
            model_path = out_dir / "ground_effect_2022_2025_model.joblib"
            features_path = out_dir / "ground_effect_2022_2025_features.joblib"
            metrics = train_and_save(data_df, model_path, features_path)
            candidate = {
                "status": "trained",
                "start_year": 2022,
                "end_year": 2025,
                "as_of": as_of.isoformat(),
                "trained_at": _training_timestamp(),
                "model_path": str(model_path),
                "features_path": str(features_path),
                **metrics,
                **details,
            }
        else:
            candidate = train_era(
                2022, 2025, "ground_effect_2022_2025", as_of, out_dir
            )
        results["ground_effect_2022_2025"] = _merge_training_result(
            existing_results.get("ground_effect_2022_2025", {}),
            candidate,
        )
    if args.mode in {"both", "active_aero"}:
        # PASSING 2024-2025 AS PRIOR FOR 2026+ (Hybrid Cold Start)
        print("\n--- Phase 2: Training Active Aero (2026-2030) with Heavy Physics Prior ---")
        
        if args.processed_data_dir:
            data_2026, data_prior, details_2026 = (
                load_persistent_active_aero_data(Path(args.processed_data_dir))
            )
            data_df, sampled_prior_rows = build_active_aero_training_data(
                data_2026,
                data_prior,
            )
            print(
                f"  Blended {sampled_prior_rows} prior laps into "
                f"{len(data_2026)} real 2026 laps from the persistent store."
            )
        else:
            # 1. Collect standard 2026 data
            data_2026, details_2026 = collect_era_data(2026, 2030, as_of)

            # 2. Collect 'Physics Prior' from 2024-2025 (50% sample)
            print("  Collecting Heavy Physics Prior from 2024-2025...")
            data_prior, _ = collect_era_data(2024, 2025, as_of)
            if not data_prior.empty:
                # Increase sample to 50% for stronger hierarchy enforcement
                data_prior = data_prior.sample(frac=0.50, random_state=42)
                data_prior['SampleWeight'] = 1.0  # Equal weight to ensure hierarchy is respected

                # Combine
                data_df = pd.concat([data_2026, data_prior], ignore_index=True)

                # 3. Explicit Upsampling of Wet/Intermediate conditions
                # If we don't have enough wet laps, the model ignores the IsWet flag.
                wet_laps = data_df[data_df['IsWet'] == 1]
                if not wet_laps.empty and len(wet_laps) < 500:
                    print(f"  Upsampling wet/inter conditions (found {len(wet_laps)} laps)...")
                    data_df = pd.concat([data_df, wet_laps, wet_laps], ignore_index=True)

                print(f"  Blended {len(data_prior)} prior laps into {len(data_2026)} real 2026 laps.")
            else:
                data_df = data_2026
            
        if data_df.empty:
            candidate = {
                "status": "no_data",
                "start_year": 2026,
                "end_year": 2030,
                "as_of": as_of.isoformat(),
                **details_2026,
            }
        else:
            model_path = out_dir / "active_aero_2026_2030_model.joblib"
            features_path = out_dir / "active_aero_2026_2030_features.joblib"
            metrics = train_and_save(
                data_df,
                model_path,
                features_path,
                validation_test_start_year=2026,
            )
            candidate = {
                "status": "trained_hybrid",
                "start_year": 2026,
                "end_year": 2030,
                "as_of": as_of.isoformat(),
                "trained_at": _training_timestamp(),
                "model_path": str(model_path),
                "features_path": str(features_path),
                **metrics,
                **details_2026
            }
        results["active_aero_2026_2030"] = _merge_training_result(
            existing_results.get("active_aero_2026_2030", {}),
            candidate,
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved metadata: {metadata_path}")

if __name__ == "__main__":
    main()
