"""
tire_life_analysis.py — Robust Tire Life Engine
================================================
Replaces simple gradient/kneedle cliff detection with:
  1. Lap-time smoothing (Savitzky-Golay or LOWESS)
  2. Statistical change-point detection (ruptures PELT)
  3. Marginal cost vs pit-stop cost optimization

The optimal tire life is NOT just the first detected "cliff."
It is the lap where the cost of staying out on old tires
exceeds the expected cost of pitting.
"""

import numpy as np
from scipy.signal import savgol_filter

# --------------------------------------------------------------------------- #
#  Default configuration — all values are overridable per-call                 #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "pit_loss_sec": 22.0,           # Fallback pit loss if track-specific data unavailable
    "fuel_burn_sec_per_lap": 0.07,  # Fuel mass effect on lap time
    "sc_aging_reduction": 0.70,     # Safety-car laps count as 70% of normal wear
    "min_stint_laps": 8,            # Minimum viable stint length
    "max_stint_laps": 50,           # Hard ceiling for any stint
    "smoothing_method": "savgol",   # "savgol" or "lowess"
    "smoothing_window": 7,          # Window size for Savitzky-Golay (must be odd)
    "smoothing_polyorder": 2,       # Polynomial order for Savitzky-Golay
    "changepoint_penalty": 3,       # PELT penalty — higher = fewer change points
    "changepoint_min_size": 5,      # Minimum segment size for change-point detection
}


# --------------------------------------------------------------------------- #
#  1. Smoothing                                                                #
# --------------------------------------------------------------------------- #
def smooth_lap_times(raw_times: list | np.ndarray, config: dict = None) -> np.ndarray:
    """
    Smooths raw lap times to reduce lap-to-lap noise while preserving
    the general degradation trend.

    Parameters
    ----------
    raw_times : array-like
        Absolute lap times in seconds (one value per lap).
    config : dict, optional
        Override defaults for smoothing_method, smoothing_window, etc.

    Returns
    -------
    np.ndarray
        Smoothed lap times (same length as input).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(raw_times, dtype=float)

    if len(arr) < 5:
        # Too few points to smooth meaningfully
        return arr.copy()

    method = cfg["smoothing_method"]

    if method == "lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            # lowess returns (x, y) sorted by x
            frac = min(0.3, max(0.1, cfg["smoothing_window"] / len(arr)))
            result = lowess(arr, np.arange(len(arr)), frac=frac, return_sorted=True)
            return result[:, 1]
        except ImportError:
            # Fall back to savgol if statsmodels unavailable
            method = "savgol"

    # Default: Savitzky-Golay
    window = cfg["smoothing_window"]
    # Window must be odd and <= data length
    if window % 2 == 0:
        window += 1
    window = min(window, len(arr))
    if window % 2 == 0:
        window -= 1
    window = max(3, window)

    polyorder = min(cfg["smoothing_polyorder"], window - 1)
    smoothed = savgol_filter(arr, window_length=window, polyorder=polyorder)

    return smoothed


# --------------------------------------------------------------------------- #
#  2. Change-Point Detection                                                   #
# --------------------------------------------------------------------------- #
def detect_change_point(smoothed_times: np.ndarray, config: dict = None) -> int | None:
    """
    Uses ruptures PELT to detect where degradation shifts from a stable
    regime to an accelerated one.

    Parameters
    ----------
    smoothed_times : np.ndarray
        Smoothed lap times.
    config : dict, optional
        Override defaults for changepoint_penalty, changepoint_min_size.

    Returns
    -------
    int or None
        The lap number (1-indexed) where the regime change occurs,
        or None if no significant change point is detected.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(smoothed_times, dtype=float)

    if len(arr) < 10:
        return None

    try:
        import ruptures as rpt
    except ImportError:
        # If ruptures is not installed, skip change-point detection
        return None

    # We detect change points on the *first derivative* (lap-to-lap delta)
    # because a "cliff" manifests as a slope change, not a level shift.
    deltas = np.diff(arr)

    # PELT algorithm with RBF kernel cost for robustness
    algo = rpt.Pelt(model="rbf", min_size=cfg["changepoint_min_size"])
    algo.fit(deltas.reshape(-1, 1))

    try:
        breakpoints = algo.predict(pen=cfg["changepoint_penalty"])
    except Exception:
        return None

    # breakpoints includes len(deltas) as the final "end" marker — remove it
    breakpoints = [bp for bp in breakpoints if bp < len(deltas)]

    if not breakpoints:
        return None

    # Return the FIRST change point as a 1-indexed lap number
    # (bp is an index into deltas, which starts at lap 2's delta)
    first_bp = breakpoints[0]
    # +2 because: deltas[0] = lap2 - lap1, so index 0 corresponds to lap 2
    change_lap = first_bp + 2

    # Validate: the change point should be in a reasonable range
    if change_lap < cfg["min_stint_laps"] or change_lap > len(arr) - 2:
        return None

    # Significance filter: verify the slope AFTER the change point is
    # meaningfully steeper than BEFORE it. On very flat curves (e.g. Hard
    # tires), PELT can detect tiny noise artifacts; this prevents false cliffs.
    pre_slope = np.mean(deltas[:first_bp]) if first_bp > 0 else 0.0
    post_slope = np.mean(deltas[first_bp:]) if first_bp < len(deltas) else 0.0

    # The post-change slope must be at least 1.5x the pre-change slope
    # AND the absolute post-slope must be non-trivial (> 0.02s per lap)
    if post_slope < pre_slope * 1.5 or post_slope < 0.02:
        return None

    return int(change_lap)


# --------------------------------------------------------------------------- #
#  3. Cumulative Degradation Cost                                              #
# --------------------------------------------------------------------------- #
def estimate_degradation_cost(
    smoothed_times: np.ndarray,
    fuel_correction: bool = True,
    config: dict = None
) -> np.ndarray:
    """
    Computes the cumulative time lost from degradation compared to
    the tire's best lap (typically lap 1-3 after warm-up).

    Parameters
    ----------
    smoothed_times : np.ndarray
        Smoothed absolute lap times.
    fuel_correction : bool
        If True, subtract the expected fuel-burn improvement per lap.
    config : dict, optional
        Override fuel_burn_sec_per_lap.

    Returns
    -------
    np.ndarray
        Cumulative degradation cost at each lap (seconds).
        cumulative_cost[i] = sum of (lap_time[j] - baseline) for j in 0..i
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(smoothed_times, dtype=float)

    # Baseline: the minimum of the first 3 laps (tire warm-up zone)
    baseline = np.min(arr[:min(3, len(arr))])

    # Per-lap degradation delta relative to baseline
    per_lap_delta = arr - baseline

    if fuel_correction:
        # Fuel burn makes the car lighter (faster) over time.
        # We ADD this back because the raw lap time already includes fuel benefit,
        # so the "true" tire degradation is slightly worse than it appears.
        fuel_correction_arr = np.arange(len(arr)) * cfg["fuel_burn_sec_per_lap"]
        per_lap_delta = per_lap_delta + fuel_correction_arr

    # Ensure non-negative (early laps might dip below baseline due to warm-up)
    per_lap_delta = np.maximum(0.0, per_lap_delta)

    # Cumulative sum
    cumulative = np.cumsum(per_lap_delta)

    return cumulative


# --------------------------------------------------------------------------- #
#  4. Tire Life Recommendation                                                 #
# --------------------------------------------------------------------------- #
def recommend_tire_life(
    raw_times: list | np.ndarray,
    pit_loss: float = None,
    config: dict = None,
) -> dict:
    """
    Master function: combines smoothing, change-point detection, and
    marginal cost analysis to recommend an optimal tire life.

    The recommended life is the EARLIEST of:
      - The lap where cumulative degradation exceeds pit loss
      - The statistical change point (if detected)

    If neither signal fires, the system falls back to a cost-based estimate
    with a lower confidence flag.

    Parameters
    ----------
    raw_times : array-like
        Absolute predicted lap times (seconds), one per lap.
    pit_loss : float, optional
        Track-specific pit-stop time loss. Falls back to config default.
    config : dict, optional
        Override any DEFAULT_CONFIG value.

    Returns
    -------
    dict with keys:
        smoothed_times:         list of smoothed lap times
        change_point_lap:       int or None
        recommended_max_life:   int
        recommendation_reason:  str
        confidence:             "high" | "medium" | "low"
        drop_off_per_lap_sec:   float (baseline degradation rate)
        cumulative_deg_cost:    list of floats
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    effective_pit_loss = pit_loss if pit_loss is not None else cfg["pit_loss_sec"]

    raw = np.asarray(raw_times, dtype=float)
    n = len(raw)

    # ---- Step 1: Smooth ----
    smoothed = smooth_lap_times(raw, cfg)

    # ---- Step 2: Change-point detection ----
    cp_lap = detect_change_point(smoothed, cfg)

    # ---- Step 3: Cumulative degradation cost ----
    cum_cost = estimate_degradation_cost(smoothed, fuel_correction=True, config=cfg)

    # ---- Step 4: Find the lap where cumulative cost exceeds pit loss ----
    # This is the "marginal cost" crossover: staying out is now worse than pitting.
    cost_crossover_lap = None
    for i in range(len(cum_cost)):
        if cum_cost[i] >= effective_pit_loss:
            cost_crossover_lap = i + 1  # 1-indexed
            break

    # ---- Step 5: Baseline degradation rate (early stint slope) ----
    eval_start = min(2, n - 1)
    eval_end = min(10, n - 1)
    if eval_end > eval_start:
        baseline_slope = (smoothed[eval_end] - smoothed[eval_start]) / (eval_end - eval_start)
    else:
        baseline_slope = 0.0
    baseline_slope = max(0.001, float(baseline_slope))

    # ---- Step 6: Decision logic ----
    candidates = []
    reasons = []

    if cp_lap is not None:
        candidates.append(cp_lap)
        reasons.append("change point detected")

    if cost_crossover_lap is not None:
        candidates.append(cost_crossover_lap)
        reasons.append("cumulative degradation exceeds pit loss")

    if candidates:
        # Take the EARLIEST signal — conservative and race-safe
        best_idx = int(np.argmin(candidates))
        recommended = candidates[best_idx]
        reason = reasons[best_idx]

        # Confidence based on signal agreement
        if len(candidates) == 2 and abs(candidates[0] - candidates[1]) <= 3:
            confidence = "high"   # Both methods agree within 3 laps
        elif len(candidates) == 2:
            confidence = "medium" # Both methods fired but disagree
        else:
            confidence = "medium" # Only one method fired
    else:
        # No clear cliff or cost crossover — fall back to 80% of race distance
        recommended = int(n * 0.80)
        reason = "no clear degradation cliff"
        confidence = "low"

    # Apply hard bounds
    recommended = max(cfg["min_stint_laps"], min(recommended, cfg["max_stint_laps"], n - 2))

    return {
        "smoothed_times": smoothed.tolist(),
        "change_point_lap": cp_lap,
        "cost_crossover_lap": cost_crossover_lap,
        "recommended_max_life": int(recommended),
        "recommendation_reason": reason,
        "confidence": confidence,
        "drop_off_per_lap_sec": round(baseline_slope, 4),
        "cumulative_deg_cost": cum_cost.tolist(),
    }
