from itertools import product
from typing import Dict, Iterable, List, Tuple


DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]


def _compound_pool(race_condition: str) -> List[str]:
    if race_condition == "wet":
        return WET_COMPOUNDS
    if race_condition == "mixed":
        return DRY_COMPOUNDS + WET_COMPOUNDS
    return DRY_COMPOUNDS


def _valid_sequence(seq: Tuple[str, ...], race_condition: str) -> bool:
    if race_condition == "dry":
        used_dry = [c for c in seq if c in DRY_COMPOUNDS]
        return len(set(used_dry)) >= 2
    return True


def _length_ranges(seq: Tuple[str, ...], models: Dict[str, Dict[str, float]]) -> List[Tuple[int, int]]:
    ranges = []
    for compound in seq:
        window = int(round(models[compound]["window_laps"]))
        lo = max(5, window - 6)
        hi = max(lo, window + 6)
        ranges.append((lo, hi))
    return ranges


def _enumerate_lengths(
    ranges: List[Tuple[int, int]],
    total_laps: int,
    step: int = 2,
) -> Iterable[Tuple[int, ...]]:
    n = len(ranges)
    if n == 1:
        lo, hi = ranges[0]
        if lo <= total_laps <= hi:
            yield (total_laps,)
        return

    def rec(idx: int, used: List[int], remaining: int):
        if idx == n - 1:
            lo, hi = ranges[idx]
            if lo <= remaining <= hi:
                yield tuple(used + [remaining])
            return
        lo, hi = ranges[idx]
        for length in range(lo, hi + 1, step):
            if length >= remaining:
                continue
            yield from rec(idx + 1, used + [length], remaining - length)

    yield from rec(0, [], total_laps)


def _score_strategy(
    seq: Tuple[str, ...],
    lengths: Tuple[int, ...],
    models: Dict[str, Dict[str, float]],
    track_length_km: float,
    pit_loss_sec: float,
) -> float:
    total = 0.0
    for compound, stint_laps in zip(seq, lengths):
        model = models[compound]
        fresh = model["fresh_lap_time_sec"]
        slope_per_lap = model["slope_sec_per_km"] * track_length_km
        for lap_idx in range(stint_laps):
            total += fresh + slope_per_lap * lap_idx
    total += pit_loss_sec * (len(seq) - 1)
    return total


def optimize_strategy(
    compound_models: Dict[str, Dict[str, float]],
    race_laps: int,
    track_length_km: float,
    race_condition: str = "dry",
    pit_loss_sec: float = 21.0,
    max_stops: int = 2,
    top_k: int = 5,
) -> List[Dict]:
    available = [c for c in _compound_pool(race_condition) if c in compound_models]
    if len(available) < 2 and race_condition == "dry":
        return []
    if not available:
        return []

    candidates: List[Dict] = []
    max_stints = max_stops + 1
    for stints in range(2, max_stints + 1):
        for seq in product(available, repeat=stints):
            if not _valid_sequence(seq, race_condition):
                continue
            ranges = _length_ranges(seq, compound_models)
            for lengths in _enumerate_lengths(ranges, race_laps):
                time_sec = _score_strategy(seq, lengths, compound_models, track_length_km, pit_loss_sec)
                candidates.append(
                    {
                        "compounds": list(seq),
                        "stint_laps": list(lengths),
                        "stops": stints - 1,
                        "predicted_total_time_sec": round(float(time_sec), 3),
                    }
                )

    candidates.sort(key=lambda x: x["predicted_total_time_sec"])
    return candidates[:top_k]
