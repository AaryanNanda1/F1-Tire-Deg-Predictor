import argparse
import json
from typing import Dict

from historical_data import build_weighted_history, resolve_target_context
from mappings import normalize_team_name
from strategy_optimizer import optimize_strategy
from tire_modeling import build_compound_models, build_overstay_table, compute_wet_experience_km


def _infer_condition(history, threshold: float = 0.25) -> str:
    if history.empty:
        return "dry"
    wet_share = float((history["IsWet"] == 1).mean())
    return "wet" if wet_share >= threshold else "dry"


def main():
    parser = argparse.ArgumentParser(description="Phased F1 tire strategy planner")
    parser.add_argument("--year", type=int, required=True, help="Target season year")
    parser.add_argument("--gp", type=str, required=True, help="Target grand prix")
    parser.add_argument("--driver", type=str, required=True, help="Driver code (e.g. VER)")
    parser.add_argument("--team", type=str, required=True, help="Team name")
    parser.add_argument("--race-laps", type=int, required=True, help="Planned race lap count")
    parser.add_argument("--condition", choices=["auto", "dry", "wet", "mixed"], default="auto")
    parser.add_argument("--pit-loss-sec", type=float, default=21.0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    team = normalize_team_name(args.team)
    target_context = resolve_target_context(args.year, args.gp)
    track_type = target_context["track_type"]
    track_length_km = target_context["track_length_km"]

    history = build_weighted_history(args.year, args.gp)
    if history.empty:
        raise SystemExit("No historical race data could be loaded for this configuration.")

    wet_experience_km = compute_wet_experience_km(
        history_df=history,
        target_year=args.year,
        driver=args.driver,
        team=team,
        target_track_type=track_type,
    )
    compound_models = build_compound_models(
        history_df=history,
        driver=args.driver,
        team=team,
        track_type=track_type,
        track_length_km=track_length_km,
        wet_experience_km=wet_experience_km,
    )
    if not compound_models:
        raise SystemExit("Unable to build compound models from available history.")

    condition = args.condition
    if condition == "auto":
        condition = _infer_condition(history)

    best = optimize_strategy(
        compound_models=compound_models,
        race_laps=args.race_laps,
        track_length_km=track_length_km,
        race_condition=condition,
        pit_loss_sec=args.pit_loss_sec,
    )
    overstay = build_overstay_table(compound_models, track_length_km)

    result: Dict = {
        "target": {
            "year": args.year,
            "grand_prix": args.gp,
            "event_name": target_context["event_name"],
            "driver": args.driver,
            "team": team,
            "track_type": track_type,
            "track_length_km": track_length_km,
            "race_laps": args.race_laps,
            "race_condition": condition,
        },
        "phase_1_history_rows": int(len(history)),
        "phase_2_compound_models": compound_models,
        "phase_2_wet_experience_km": round(wet_experience_km, 3),
        "phase_3_best_strategies": best,
        "phase_3_overstay_delta": overstay,
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
