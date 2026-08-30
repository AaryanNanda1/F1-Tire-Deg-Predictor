"""Race-level strategy and tire-life benchmarking utilities.

The benchmark deliberately treats Pirelli recommendations as an expert baseline
and observed race strategies as contextual outcomes. It does not use either as
training labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import csv
import json


VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
MODEL_STRATEGY_KEYS = (
    ("best_strategy", "optimal"),
    ("safe_strategy", "safe"),
    ("risky_strategy", "risky"),
)


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark or prediction suite is structurally invalid."""


@dataclass(frozen=True)
class NormalizedStrategy:
    label: str
    stops: int
    compounds: List[str]
    stint_laps: List[int]
    pit_laps: List[int]


def load_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkValidationError(message)


def _validate_strategy(strategy: Mapping[str, Any], context: str) -> None:
    stop_count = strategy.get("stop_count")
    _require(
        isinstance(stop_count, int) and stop_count >= 0,
        f"{context}.stop_count must be a non-negative integer",
    )

    compounds = strategy.get("compounds")
    if compounds is not None:
        _require(isinstance(compounds, list), f"{context}.compounds must be a list or null")
        normalized = [str(value).upper() for value in compounds]
        _require(
            all(value in VALID_COMPOUNDS for value in normalized),
            f"{context}.compounds contains an unsupported compound",
        )
        _require(
            len(normalized) == stop_count + 1,
            f"{context}.compounds must contain stop_count + 1 stints",
        )

    stint_laps = strategy.get("stint_laps")
    if stint_laps is not None:
        _require(isinstance(stint_laps, list), f"{context}.stint_laps must be a list or null")
        _require(
            all(isinstance(value, int) and value > 0 for value in stint_laps),
            f"{context}.stint_laps must contain positive integers",
        )
        if compounds is not None:
            _require(
                len(stint_laps) == len(compounds),
                f"{context}.stint_laps must align with compounds",
            )

    pit_windows = strategy.get("pit_windows")
    if pit_windows is not None:
        _require(isinstance(pit_windows, list), f"{context}.pit_windows must be a list or null")
        _require(
            len(pit_windows) == stop_count,
            f"{context}.pit_windows must contain one window per stop",
        )
        for index, window in enumerate(pit_windows):
            _require(
                isinstance(window, list)
                and len(window) == 2
                and all(isinstance(value, int) for value in window)
                and window[0] <= window[1],
                f"{context}.pit_windows[{index}] must be [minimum_lap, maximum_lap]",
            )


def validate_benchmark_suite(suite: Mapping[str, Any]) -> None:
    _require(suite.get("schema_version") == 1, "benchmark schema_version must be 1")
    races = suite.get("races")
    _require(isinstance(races, list) and races, "benchmark races must be a non-empty list")

    seen_ids = set()
    for index, race in enumerate(races):
        context = f"races[{index}]"
        _require(isinstance(race, dict), f"{context} must be an object")
        race_id = race.get("id")
        _require(isinstance(race_id, str) and race_id, f"{context}.id is required")
        _require(race_id not in seen_ids, f"duplicate race id: {race_id}")
        seen_ids.add(race_id)
        _require(
            race.get("split") in {"calibration", "holdout"},
            f"{context}.split must be calibration or holdout",
        )
        _require(
            isinstance(race.get("season"), int),
            f"{context}.season must be an integer",
        )
        _require(
            isinstance(race.get("event_name"), str) and race["event_name"],
            f"{context}.event_name is required",
        )
        _require(
            isinstance(race.get("race_laps"), int) and race["race_laps"] > 0,
            f"{context}.race_laps must be positive",
        )
        _require(
            isinstance(race.get("track_name"), str) and race["track_name"],
            f"{context}.track_name is required",
        )
        simulation_input = race.get("simulation_input")
        _require(
            isinstance(simulation_input, dict),
            f"{context}.simulation_input must be an object",
        )
        for field in ("driver", "team", "race_date"):
            _require(
                isinstance(simulation_input.get(field), str)
                and simulation_input[field],
                f"{context}.simulation_input.{field} is required",
            )

        pirelli = race.get("pirelli")
        observed = race.get("observed")
        _require(isinstance(pirelli, dict), f"{context}.pirelli must be an object")
        _require(isinstance(observed, dict), f"{context}.observed must be an object")
        _require(
            str(pirelli.get("source_url", "")).startswith("https://press.pirelli.com/"),
            f"{context}.pirelli.source_url must reference Pirelli",
        )
        _require(
            str(observed.get("source_url", "")).startswith("https://press.pirelli.com/"),
            f"{context}.observed.source_url must reference Pirelli",
        )

        compound_mapping = pirelli.get("compound_mapping", {})
        _require(isinstance(compound_mapping, dict), f"{context}.compound_mapping must be an object")
        for relative, specification in compound_mapping.items():
            _require(relative in VALID_COMPOUNDS, f"{context} has invalid relative compound {relative}")
            _require(
                isinstance(specification, str)
                and specification in {"C1", "C2", "C3", "C4", "C5"},
                f"{context} has invalid Pirelli specification {specification}",
            )

        recommendations = pirelli.get("recommended_strategies", [])
        references = observed.get("reference_strategies", [])
        _require(isinstance(recommendations, list), f"{context}.recommended_strategies must be a list")
        _require(isinstance(references, list), f"{context}.reference_strategies must be a list")
        for strategy_index, strategy in enumerate(recommendations):
            _validate_strategy(strategy, f"{context}.recommended_strategies[{strategy_index}]")
        for strategy_index, strategy in enumerate(references):
            _validate_strategy(strategy, f"{context}.reference_strategies[{strategy_index}]")

        longest_stints = observed.get("longest_stints", {})
        _require(isinstance(longest_stints, dict), f"{context}.longest_stints must be an object")
        for compound, stint in longest_stints.items():
            _require(compound in VALID_COMPOUNDS, f"{context} has invalid longest-stint compound")
            _require(isinstance(stint, dict), f"{context}.longest_stints.{compound} must be an object")
            _require(
                isinstance(stint.get("laps"), int) and stint["laps"] > 0,
                f"{context}.longest_stints.{compound}.laps must be positive",
            )


def validate_prediction_suite(predictions: Mapping[str, Any]) -> None:
    _require(predictions.get("schema_version") == 1, "prediction schema_version must be 1")
    _require(
        predictions.get("evaluation_mode") in {"production_diagnostic", "leakage_free"},
        "evaluation_mode must be production_diagnostic or leakage_free",
    )
    races = predictions.get("races")
    _require(isinstance(races, dict), "prediction races must be an object keyed by race id")


def _normalize_compounds(values: Iterable[Any]) -> List[str]:
    return [str(value).upper() for value in values]


def _normalize_model_strategy(strategy: Mapping[str, Any], label: str) -> NormalizedStrategy:
    stints_data = strategy.get("stints_data") or []
    if stints_data:
        compounds = _normalize_compounds(stint["compound"] for stint in stints_data)
        stint_laps = [int(stint["laps"]) for stint in stints_data]
    else:
        compounds = _normalize_compounds(strategy.get("compounds") or [])
        stint_laps = [int(value) for value in strategy.get("stint_laps") or strategy.get("stints") or []]

    stops = int(strategy.get("stops", max(0, len(compounds) - 1)))
    pit_laps = []
    cumulative = 0
    for stint_length in stint_laps[:-1]:
        cumulative += stint_length
        pit_laps.append(cumulative)
    return NormalizedStrategy(label, stops, compounds, stint_laps, pit_laps)


def extract_model_strategies(model_output: Mapping[str, Any]) -> List[NormalizedStrategy]:
    strategy_block = model_output.get("strategies", model_output)
    strategies = []
    for key, label in MODEL_STRATEGY_KEYS:
        value = strategy_block.get(key)
        if value:
            strategies.append(_normalize_model_strategy(value, label))
    return strategies


def extract_useful_life(model_output: Mapping[str, Any]) -> Dict[str, int]:
    compound_block = model_output.get("degradation_graphs") or model_output.get("compounds") or {}
    useful_life = {}
    for compound, values in compound_block.items():
        lap = values.get("strategy_useful_life_lap")
        if lap is not None:
            useful_life[str(compound).upper()] = int(lap)
    return useful_life


def extract_compound_diagnostics(
    model_output: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    compound_block = (
        model_output.get("degradation_graphs")
        or model_output.get("compounds")
        or {}
    )
    return {
        str(compound).upper(): values
        for compound, values in compound_block.items()
    }


def _stop_match(strategy: NormalizedStrategy, references: Sequence[Mapping[str, Any]]) -> Optional[bool]:
    if not references:
        return None
    return any(strategy.stops == int(reference["stop_count"]) for reference in references)


def _sequence_match(
    strategy: NormalizedStrategy,
    references: Sequence[Mapping[str, Any]],
    *,
    order_sensitive: bool,
) -> Optional[bool]:
    available = [reference for reference in references if reference.get("compounds") is not None]
    if not available or not strategy.compounds:
        return None
    predicted = strategy.compounds if order_sensitive else Counter(strategy.compounds)
    for reference in available:
        expected_values = _normalize_compounds(reference["compounds"])
        expected = expected_values if order_sensitive else Counter(expected_values)
        if predicted == expected:
            return True
    return False


def _distance_to_window(lap: int, window: Sequence[int]) -> int:
    low, high = int(window[0]), int(window[1])
    if lap < low:
        return low - lap
    if lap > high:
        return lap - high
    return 0


def _pit_window_mae(
    strategy: NormalizedStrategy,
    references: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    available = [
        reference
        for reference in references
        if reference.get("pit_windows") is not None
        and len(reference["pit_windows"]) == len(strategy.pit_laps)
    ]
    if not available or not strategy.pit_laps:
        return None
    candidate_errors = []
    for reference in available:
        errors = [
            _distance_to_window(lap, window)
            for lap, window in zip(strategy.pit_laps, reference["pit_windows"])
        ]
        candidate_errors.append(mean(errors))
    return round(min(candidate_errors), 3)


def _any_match(values: Sequence[Optional[bool]]) -> Optional[bool]:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return any(available)


def _evaluate_strategy_references(
    strategies: Sequence[NormalizedStrategy],
    references: Sequence[Mapping[str, Any]],
    prefix: str,
) -> Dict[str, Optional[float | bool]]:
    best = strategies[0] if strategies else None
    if not best:
        return {
            f"best_{prefix}_stop_count_match": None,
            f"top_three_{prefix}_stop_count_match": None,
            f"best_{prefix}_exact_sequence_match": None,
            f"top_three_{prefix}_exact_sequence_match": None,
            f"best_{prefix}_compound_set_match": None,
            f"top_three_{prefix}_compound_set_match": None,
            f"best_{prefix}_pit_window_mae_laps": None,
        }

    stop_matches = [_stop_match(strategy, references) for strategy in strategies]
    exact_matches = [
        _sequence_match(strategy, references, order_sensitive=True)
        for strategy in strategies
    ]
    set_matches = [
        _sequence_match(strategy, references, order_sensitive=False)
        for strategy in strategies
    ]
    return {
        f"best_{prefix}_stop_count_match": stop_matches[0],
        f"top_three_{prefix}_stop_count_match": _any_match(stop_matches),
        f"best_{prefix}_exact_sequence_match": exact_matches[0],
        f"top_three_{prefix}_exact_sequence_match": _any_match(exact_matches),
        f"best_{prefix}_compound_set_match": set_matches[0],
        f"top_three_{prefix}_compound_set_match": _any_match(set_matches),
        f"best_{prefix}_pit_window_mae_laps": _pit_window_mae(best, references),
    }


def evaluate_race(
    benchmark_race: Mapping[str, Any],
    model_output: Mapping[str, Any],
) -> Dict[str, Any]:
    strategies = extract_model_strategies(model_output)
    useful_life = extract_useful_life(model_output)
    compound_diagnostics = extract_compound_diagnostics(model_output)
    strategy_block = model_output.get("strategies", model_output)
    raw_strategies = {
        label: strategy_block.get(key) or {}
        for key, label in MODEL_STRATEGY_KEYS
    }
    pirelli_references = benchmark_race["pirelli"].get("recommended_strategies", [])
    actual_references = benchmark_race["observed"].get("reference_strategies", [])

    strategy_metrics = {}
    strategy_metrics.update(
        _evaluate_strategy_references(strategies, pirelli_references, "pirelli")
    )
    strategy_metrics.update(
        _evaluate_strategy_references(strategies, actual_references, "actual")
    )

    tire_life_comparisons = []
    for compound, observed in benchmark_race["observed"].get("longest_stints", {}).items():
        if compound not in useful_life:
            continue
        predicted_laps = useful_life[compound]
        observed_laps = int(observed["laps"])
        demonstrated_margin = predicted_laps - observed_laps
        demonstrated_shortfall = max(0, -demonstrated_margin)
        compound_values = compound_diagnostics.get(compound, {})
        performance_cliff = compound_values.get("performance_cliff_lap")
        tire_life_comparisons.append(
            {
                "compound": compound,
                "pirelli_specification": benchmark_race["pirelli"]
                .get("compound_mapping", {})
                .get(compound),
                "predicted_useful_life_laps": predicted_laps,
                "demonstrated_longest_stint_laps": observed_laps,
                "observed_driver": observed.get("driver"),
                "useful_life_margin_vs_demonstrated_laps": demonstrated_margin,
                "useful_life_below_demonstrated_laps": demonstrated_shortfall,
                "useful_life_at_or_above_demonstrated": (
                    demonstrated_margin >= 0
                ),
                "predicted_performance_cliff_lap": performance_cliff,
                "performance_cliff_confidence": compound_values.get(
                    "cliff_confidence"
                ),
                "performance_cliff_method": compound_values.get(
                    "cliff_method"
                ),
                "observed_stint_margin_beyond_cliff_laps": (
                    None
                    if performance_cliff is None
                    else observed_laps - int(performance_cliff)
                ),
                "predicted_cliff_before_demonstrated_stint": (
                    None
                    if performance_cliff is None
                    else int(performance_cliff) < observed_laps
                ),
            }
        )

    return {
        "race_id": benchmark_race["id"],
        "split": benchmark_race["split"],
        "season": benchmark_race["season"],
        "event_name": benchmark_race["event_name"],
        "race_interruptions": benchmark_race["observed"].get("race_interruptions", []),
        "weather_context": (
            model_output.get("input_context", {}).get("weather")
            or model_output.get("weather_forecast")
        ),
        "model_strategies": [
            {
                "label": strategy.label,
                "stops": strategy.stops,
                "compounds": strategy.compounds,
                "stint_laps": strategy.stint_laps,
                "pit_laps": strategy.pit_laps,
                "expected_total_time_sec": raw_strategies[strategy.label].get(
                    "expected_total_time_sec"
                ),
                "risk_adjusted_total_time_sec": raw_strategies[
                    strategy.label
                ].get("risk_adjusted_total_time_sec"),
                "cost_breakdown": raw_strategies[strategy.label].get(
                    "cost_breakdown"
                ),
                "max_performance_cliff_overshoot_laps": raw_strategies[
                    strategy.label
                ].get("max_performance_cliff_overshoot_laps"),
                "max_useful_life_overshoot_laps": raw_strategies[
                    strategy.label
                ].get("max_useful_life_overshoot_laps"),
            }
            for strategy in strategies
        ],
        "strategy_metrics": strategy_metrics,
        "tire_life_comparisons": tire_life_comparisons,
        "sources": {
            "pirelli": benchmark_race["pirelli"]["source_url"],
            "observed": benchmark_race["observed"]["source_url"],
        },
    }


def _rate(reports: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    values = [
        report["strategy_metrics"].get(metric)
        for report in reports
        if report["strategy_metrics"].get(metric) is not None
    ]
    if not values:
        return None
    return round(sum(bool(value) for value in values) / len(values), 3)


def _available_count(reports: Sequence[Mapping[str, Any]], metric: str) -> int:
    return sum(
        report["strategy_metrics"].get(metric) is not None
        for report in reports
    )


def _numeric_mean(reports: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    values = [
        float(report["strategy_metrics"][metric])
        for report in reports
        if report["strategy_metrics"].get(metric) is not None
    ]
    return round(mean(values), 3) if values else None


def aggregate_reports(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    boolean_metrics = (
        "best_pirelli_stop_count_match",
        "top_three_pirelli_stop_count_match",
        "best_pirelli_exact_sequence_match",
        "top_three_pirelli_exact_sequence_match",
        "best_pirelli_compound_set_match",
        "top_three_pirelli_compound_set_match",
        "best_actual_stop_count_match",
        "top_three_actual_stop_count_match",
        "best_actual_exact_sequence_match",
        "top_three_actual_exact_sequence_match",
        "best_actual_compound_set_match",
        "top_three_actual_compound_set_match",
    )
    aggregate = {}
    for metric in boolean_metrics:
        aggregate[f"{metric}_rate"] = _rate(reports, metric)
        aggregate[f"{metric}_observations"] = _available_count(reports, metric)
    for metric in (
        "best_pirelli_pit_window_mae_laps",
        "best_actual_pit_window_mae_laps",
    ):
        aggregate[metric] = _numeric_mean(reports, metric)
        aggregate[f"{metric}_observations"] = _available_count(reports, metric)

    life_rows = [
        row
        for report in reports
        for row in report.get("tire_life_comparisons", [])
    ]
    if life_rows:
        demonstrated_margins = [
            row["useful_life_margin_vs_demonstrated_laps"]
            for row in life_rows
        ]
        demonstrated_shortfalls = [
            row["useful_life_below_demonstrated_laps"]
            for row in life_rows
            if row["useful_life_below_demonstrated_laps"] > 0
        ]
        aggregate.update(
            {
                "demonstrated_stint_observations": len(life_rows),
                "useful_life_at_or_above_demonstrated_rate": round(
                    sum(
                        row["useful_life_at_or_above_demonstrated"]
                        for row in life_rows
                    )
                    / len(life_rows),
                    3,
                ),
                "useful_life_below_demonstrated_rate": round(
                    len(demonstrated_shortfalls) / len(life_rows),
                    3,
                ),
                "useful_life_below_demonstrated_mean_shortfall_laps": (
                    round(mean(demonstrated_shortfalls), 3)
                    if demonstrated_shortfalls
                    else 0.0
                ),
                "useful_life_margin_vs_demonstrated_mean_laps": round(
                    mean(demonstrated_margins),
                    3,
                ),
            }
        )
        cliff_margins = [
            row["observed_stint_margin_beyond_cliff_laps"]
            for row in life_rows
            if row.get("observed_stint_margin_beyond_cliff_laps") is not None
        ]
        cliff_before_demonstrated = [
            row["predicted_cliff_before_demonstrated_stint"]
            for row in life_rows
            if row.get("predicted_cliff_before_demonstrated_stint") is not None
        ]
        aggregate["performance_cliff_context_observations"] = len(
            cliff_margins
        )
        aggregate["observed_stint_margin_beyond_cliff_mean_laps"] = (
            round(mean(cliff_margins), 3) if cliff_margins else None
        )
        aggregate["predicted_cliff_before_demonstrated_stint_rate"] = (
            round(
                sum(cliff_before_demonstrated)
                / len(cliff_before_demonstrated),
                3,
            )
            if cliff_before_demonstrated
            else None
        )
    else:
        aggregate.update(
            {
                "demonstrated_stint_observations": 0,
                "useful_life_at_or_above_demonstrated_rate": None,
                "useful_life_below_demonstrated_rate": None,
                "useful_life_below_demonstrated_mean_shortfall_laps": None,
                "useful_life_margin_vs_demonstrated_mean_laps": None,
                "performance_cliff_context_observations": 0,
                "observed_stint_margin_beyond_cliff_mean_laps": None,
                "predicted_cliff_before_demonstrated_stint_rate": None,
            }
        )
    return aggregate


def evaluate_suite(
    benchmark_suite: Mapping[str, Any],
    prediction_suite: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_benchmark_suite(benchmark_suite)
    validate_prediction_suite(prediction_suite)

    prediction_races = prediction_suite["races"]
    reports = []
    missing = []
    for benchmark_race in benchmark_suite["races"]:
        race_id = benchmark_race["id"]
        if race_id not in prediction_races:
            missing.append(race_id)
            continue
        reports.append(evaluate_race(benchmark_race, prediction_races[race_id]))

    _require(not missing, f"missing predictions for: {', '.join(missing)}")
    mode = prediction_suite["evaluation_mode"]
    reports_by_split = {
        split: [report for report in reports if report["split"] == split]
        for split in ("calibration", "holdout")
    }
    return {
        "schema_version": 1,
        "suite_name": benchmark_suite.get("suite_name", "Pirelli strategy benchmark"),
        "evaluation_mode": mode,
        "leakage_warning": (
            None
            if mode == "leakage_free"
            else "Production artifacts may contain data from benchmark races; use these results diagnostically."
        ),
        "model_training_cutoff": prediction_suite.get("model_training_cutoff"),
        "model_artifacts": prediction_suite.get("model_artifacts"),
        "benchmark_training_overlap": prediction_suite.get(
            "benchmark_training_overlap"
        ),
        "capture_assumptions": prediction_suite.get("capture_assumptions", []),
        "race_count": len(reports),
        "aggregate_metrics": aggregate_reports(reports),
        "aggregate_metrics_by_split": {
            split: aggregate_reports(split_reports)
            for split, split_reports in reports_by_split.items()
            if split_reports
        },
        "races": reports,
    }


def write_report_json(report: Mapping[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def write_report_csv(report: Mapping[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for race in report["races"]:
        base = {
            "race_id": race["race_id"],
            "split": race["split"],
            "season": race["season"],
            "event_name": race["event_name"],
            "race_interruptions": ",".join(race["race_interruptions"]),
        }
        base.update(race["strategy_metrics"])
        if race["tire_life_comparisons"]:
            for comparison in race["tire_life_comparisons"]:
                row = dict(base)
                row.update(comparison)
                rows.append(row)
        else:
            rows.append(base)

    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report_markdown(report: Mapping[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate = report["aggregate_metrics"]
    lines = [
        f"# {report['suite_name']}",
        "",
        f"- Evaluation mode: `{report['evaluation_mode']}`",
        f"- Races evaluated: {report['race_count']}",
    ]
    if report.get("leakage_warning"):
        lines.append(f"- Warning: {report['leakage_warning']}")
    for assumption in report.get("capture_assumptions", []):
        lines.append(f"- Capture assumption: {assumption}")
    for race_id, overlap in (
        report.get("benchmark_training_overlap") or {}
    ).items():
        matches = overlap.get("matched_training_events", [])
        lines.append(
            f"- Training overlap for `{race_id}`: "
            f"{', '.join(matches) if matches else 'none recorded'}"
        )
    lines.extend(["", "## Aggregate metrics", ""])
    for key, value in aggregate.items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Race results", ""])
    for race in report["races"]:
        lines.extend(
            [
                f"### {race['season']} {race['event_name']}",
                "",
                f"- Interruptions: {', '.join(race['race_interruptions']) or 'none recorded'}",
                (
                    "- Weather source: "
                    f"{(race.get('weather_context') or {}).get('source', 'not recorded')}"
                ),
                f"- Pirelli source: {race['sources']['pirelli']}",
                f"- Observed source: {race['sources']['observed']}",
                "",
            ]
        )
        for key, value in race["strategy_metrics"].items():
            lines.append(f"- `{key}`: {value}")
        for comparison in race["tire_life_comparisons"]:
            cliff_context = ""
            if comparison["predicted_performance_cliff_lap"] is not None:
                cliff_context = (
                    f", cliff {comparison['predicted_performance_cliff_lap']} "
                    f"({comparison['performance_cliff_method']}), "
                    "observed stint margin beyond cliff "
                    f"{comparison['observed_stint_margin_beyond_cliff_laps']:+d}"
                )
            lines.append(
                "- "
                f"{comparison['compound']} ({comparison['pirelli_specification']}): "
                f"predicted {comparison['predicted_useful_life_laps']} laps, "
                "demonstrated longest stint "
                f"{comparison['demonstrated_longest_stint_laps']} laps, "
                "contextual margin "
                f"{comparison['useful_life_margin_vs_demonstrated_laps']:+d}"
                f"{cliff_context}"
            )
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")
