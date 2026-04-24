"""
tire_life_analysis.py — Dual-Output Tire Life Engine
=====================================================
Produces TWO independent outputs for each tire stint:

  1. performance_cliff_lap
     "Where does the lap-time curve begin to noticeably worsen?"
     Based purely on the tire's smoothed lap-time trend.
     Uses a SUSTAINED ACCELERATION RULE: slope > threshold,
     curvature > 0, and the pattern persists for N consecutive laps.

  2. strategy_useful_life_lap
     "When does staying out become worse than pitting?"
     Based on cumulative degradation cost vs pit-stop loss.
     A tire may be past its performance cliff but still worth
     using if the pit loss is large.

These are NOT the same thing. A tire may begin degrading on lap 18
but still be strategically worth using until lap 24.
"""

import numpy as np
from scipy.signal import savgol_filter

# --------------------------------------------------------------------------- #
#  Default configuration — all values are overridable per-call                 #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    # --- Smoothing ---
    "smoothing_method": "savgol",     # "savgol" or "lowess"
    "smoothing_window": 7,            # Window size for Savitzky-Golay (must be odd)
    "smoothing_polyorder": 2,         # Polynomial order for Savitzky-Golay

    # --- Performance Cliff Detection ---
    "cliff_slope_threshold": 0.05,    # Min slope (s/lap) to consider as degradation (was 0.08)
    "cliff_curvature_threshold": 0.005,# Min 2nd derivative to confirm acceleration (was 0.02)
    "cliff_baseline_window": 3,       # Number of early laps to establish baseline
    "cliff_baseline_delta": 0.3,      # Lap time must be this much worse than baseline (s) (was 0.5)
    "cliff_persistence_laps": 2,      # Consecutive laps the rule must hold (was 3)
    "cliff_min_lap": 5,               # Earliest lap a cliff can be reported

    # --- Strategy Useful Life ---
    "pit_loss_sec": 22.0,             # Fallback pit loss if track-specific unavailable
    "fuel_burn_sec_per_lap": 0.07,    # Fuel mass effect on lap time
    "sc_aging_reduction": 0.70,       # Safety-car laps count as 70% of normal wear
    "min_stint_laps": 8,              # Minimum viable stint length
    "max_stint_laps": 50,             # Hard ceiling for any stint
}


# --------------------------------------------------------------------------- #
#  1. Smoothing (shared by both detectors)                                     #
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
        return arr.copy()

    method = cfg["smoothing_method"]

    if method == "lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            frac = min(0.3, max(0.1, cfg["smoothing_window"] / len(arr)))
            result = lowess(arr, np.arange(len(arr)), frac=frac, return_sorted=True)
            return result[:, 1]
        except ImportError:
            method = "savgol"

    # Default: Savitzky-Golay
    window = cfg["smoothing_window"]
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
#  2. PERFORMANCE CLIFF DETECTION                                              #
#     "Where does the lap-time curve begin to noticeably worsen?"              #
#                                                                              #
#     This is purely about the TIRE'S PHYSICS, not pit strategy.               #
#     It uses a SUSTAINED ACCELERATION RULE:                                   #
#       - Slope (1st derivative) exceeds a configurable threshold              #
#       - Curvature (2nd derivative) is positive (degradation accelerating)    #
#       - The pattern persists for N consecutive laps                           #
#       - Lap time is meaningfully worse than the early-stint baseline          #
# --------------------------------------------------------------------------- #
def detect_performance_cliff(
    smoothed_times: np.ndarray,
    config: dict = None,
) -> dict:
    """
    Detects the first tire-age lap where performance begins to noticeably
    and significantly worsen, based on the smoothed lap-time trend.

    This does NOT consider pit strategy — only the tire's physics.

    Parameters
    ----------
    smoothed_times : np.ndarray
        Smoothed absolute lap times.
    config : dict, optional
        Override cliff_slope_threshold, cliff_curvature_threshold, etc.

    Returns
    -------
    dict with keys:
        performance_cliff_lap:  int or None (1-indexed)
        cliff_confidence:       "high" | "medium" | "low" | None
        cliff_reason:           str
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(smoothed_times, dtype=float)
    n = len(arr)

    if n < 10:
        return {
            "performance_cliff_lap": None,
            "cliff_confidence": None,
            "cliff_reason": "insufficient data (fewer than 10 laps)",
        }

    # --- Compute derivatives ---
    # 1st derivative (slope): how fast is the lap time increasing?
    slope = np.gradient(arr)
    # 2nd derivative (curvature): is the degradation accelerating?
    curvature = np.gradient(slope)

    # --- Establish early-stint baseline ---
    bw = cfg["cliff_baseline_window"]
    baseline = np.mean(arr[:min(bw, n)])

    # --- Thresholds ---
    slope_thresh = cfg["cliff_slope_threshold"]
    curve_thresh = cfg["cliff_curvature_threshold"]
    delta_thresh = cfg["cliff_baseline_delta"]
    persist = cfg["cliff_persistence_laps"]
    min_lap = cfg["cliff_min_lap"]

    # --- Sustained acceleration scan ---
    # We look for the first lap where ALL of the following hold
    # for `persist` consecutive laps:
    #   1. slope[i] > slope_threshold  (lap time is rising fast enough)
    #   2. curvature[i] > curvature_threshold  (degradation is accelerating)
    #   3. arr[i] - baseline > baseline_delta  (meaningfully worse than fresh)
    consecutive = 0
    cliff_start = None

    for i in range(min_lap - 1, n):  # min_lap is 1-indexed; i is 0-indexed
        cond_slope = slope[i] > slope_thresh
        cond_curve = curvature[i] > curve_thresh
        cond_delta = (arr[i] - baseline) > delta_thresh

        if cond_slope and cond_curve and cond_delta:
            consecutive += 1
            if cliff_start is None:
                cliff_start = i
            if consecutive >= persist:
                # Found a sustained acceleration — report the START of the pattern
                cliff_lap = cliff_start + 1  # 1-indexed
                return {
                    "performance_cliff_lap": int(cliff_lap),
                    "cliff_confidence": "high" if consecutive >= persist + 1 else "medium",
                    "cliff_reason": (
                        f"sustained acceleration detected: slope>{slope_thresh:.2f}s/lap, "
                        f"curvature>{curve_thresh:.2f}, for {consecutive} consecutive laps"
                    ),
                }
        else:
            consecutive = 0
            cliff_start = None

    # No sustained acceleration found — this is a valid outcome
    return {
        "performance_cliff_lap": None,
        "cliff_confidence": None,
        "cliff_reason": "no clear performance cliff detected",
    }


# --------------------------------------------------------------------------- #
#  3. STRATEGY USEFUL LIFE                                                     #
#     "When does staying out become worse than pitting?"                        #
#                                                                              #
#     This is about RACE STRATEGY, not tire physics.                           #
#     A tire past its performance cliff may still be worth using               #
#     if the pit loss is large (e.g., Monaco ~19s pit loss).                   #
# --------------------------------------------------------------------------- #
def estimate_degradation_cost(
    smoothed_times: np.ndarray,
    fuel_correction: bool = True,
    config: dict = None,
) -> np.ndarray:
    """
    Computes the cumulative time lost from degradation compared to
    the tire's best lap (typically lap 1-3 after warm-up).

    Parameters
    ----------
    smoothed_times : np.ndarray
        Smoothed absolute lap times.
    fuel_correction : bool
        If True, add fuel-burn correction (the raw times already include
        the fuel-weight benefit, so we add it back to isolate tire deg).
    config : dict, optional
        Override fuel_burn_sec_per_lap.

    Returns
    -------
    np.ndarray
        Cumulative degradation cost at each lap (seconds).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(smoothed_times, dtype=float)

    baseline = np.min(arr[:min(3, len(arr))])
    per_lap_delta = arr - baseline

    if fuel_correction:
        fuel_correction_arr = np.arange(len(arr)) * cfg["fuel_burn_sec_per_lap"]
        per_lap_delta = per_lap_delta + fuel_correction_arr

    per_lap_delta = np.maximum(0.0, per_lap_delta)
    cumulative = np.cumsum(per_lap_delta)

    return cumulative


def recommend_strategy_useful_life(
    smoothed_times: np.ndarray,
    pit_loss: float = None,
    config: dict = None,
) -> dict:
    """
    Determines the last lap where staying out is still a better strategic
    decision than pitting, based on cumulative degradation cost vs pit loss.

    Parameters
    ----------
    smoothed_times : np.ndarray
        Smoothed absolute lap times.
    pit_loss : float, optional
        Track-specific pit-stop time loss. Falls back to config default.
    config : dict, optional
        Override pit_loss_sec, fuel_burn_sec_per_lap, etc.

    Returns
    -------
    dict with keys:
        strategy_useful_life_lap:   int (1-indexed)
        strategy_confidence:        "high" | "medium" | "low"
        strategy_reason:            str
        cumulative_deg_cost:        list of floats
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    effective_pit_loss = pit_loss if pit_loss is not None else cfg["pit_loss_sec"]
    arr = np.asarray(smoothed_times, dtype=float)
    n = len(arr)

    # --- Cumulative degradation cost ---
    cum_cost = estimate_degradation_cost(arr, fuel_correction=True, config=cfg)

    # --- Find the lap where cumulative cost exceeds pit loss ---
    crossover_lap = None
    for i in range(len(cum_cost)):
        if cum_cost[i] >= effective_pit_loss:
            crossover_lap = i + 1  # 1-indexed
            break

    # --- Decision logic ---
    if crossover_lap is not None:
        useful_life = crossover_lap
        reason = f"cumulative degradation ({cum_cost[crossover_lap - 1]:.1f}s) exceeds pit loss ({effective_pit_loss:.1f}s)"
        confidence = "high" if crossover_lap > cfg["min_stint_laps"] else "medium"
    else:
        # Cost never exceeds pit loss — the tire is viable for the full stint
        useful_life = min(n, cfg["max_stint_laps"])
        reason = f"cumulative degradation never exceeds pit loss ({effective_pit_loss:.1f}s)"
        confidence = "low"

    # Apply hard bounds
    useful_life = max(cfg["min_stint_laps"], min(useful_life, cfg["max_stint_laps"], n - 1))

    return {
        "strategy_useful_life_lap": int(useful_life),
        "strategy_confidence": confidence,
        "strategy_reason": reason,
        "cumulative_deg_cost": cum_cost.tolist(),
    }


# --------------------------------------------------------------------------- #
#  4. MASTER FUNCTION — combines both detectors                                #
# --------------------------------------------------------------------------- #
def analyze_tire_life(
    raw_times: list | np.ndarray,
    pit_loss: float = None,
    config: dict = None,
) -> dict:
    """
    Master function: runs smoothing, performance cliff detection, and
    strategy useful life analysis independently, returning both outputs.

    Parameters
    ----------
    raw_times : array-like
        Absolute predicted lap times (seconds), one per lap.
    pit_loss : float, optional
        Track-specific pit-stop time loss.
    config : dict, optional
        Override any DEFAULT_CONFIG value.

    Returns
    -------
    dict with keys from both detectors, plus shared data:
        smoothed_times, drop_off_per_lap_sec,
        performance_cliff_lap, cliff_confidence, cliff_reason,
        strategy_useful_life_lap, strategy_confidence, strategy_reason,
        cumulative_deg_cost
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    raw = np.asarray(raw_times, dtype=float)
    n = len(raw)

    # ---- Step 1: Smooth (shared input for both detectors) ----
    smoothed = smooth_lap_times(raw, cfg)

    # ---- Step 2: Performance cliff (tire physics) ----
    cliff_result = detect_performance_cliff(smoothed, cfg)

    # ---- Step 3: Strategy useful life (race strategy) ----
    strategy_result = recommend_strategy_useful_life(smoothed, pit_loss, cfg)

    # ---- Step 4: Baseline degradation rate ----
    eval_start = min(2, n - 1)
    eval_end = min(10, n - 1)
    if eval_end > eval_start:
        baseline_slope = (smoothed[eval_end] - smoothed[eval_start]) / (eval_end - eval_start)
    else:
        baseline_slope = 0.0
    baseline_slope = max(0.001, float(baseline_slope))

    return {
        # Shared
        "smoothed_times": smoothed.tolist(),
        "drop_off_per_lap_sec": round(baseline_slope, 4),

        # Performance cliff (tire physics)
        "performance_cliff_lap": cliff_result["performance_cliff_lap"],
        "cliff_confidence": cliff_result["cliff_confidence"],
        "cliff_reason": cliff_result["cliff_reason"],

        # Strategy useful life (race strategy)
        "strategy_useful_life_lap": strategy_result["strategy_useful_life_lap"],
        "strategy_confidence": strategy_result["strategy_confidence"],
        "strategy_reason": strategy_result["strategy_reason"],
        "cumulative_deg_cost": strategy_result["cumulative_deg_cost"],
    }
