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
    "cliff_detection_method": "sustained",  # "sustained", "piecewise", or "hybrid"

    # --- Rolling-Trend Sustained Cliff Detection ---
    "rolling_trend_window": 4,
    "rolling_min_slope_increase": 0.05,
    "rolling_min_fit_improvement_ratio": 0.20,

    # --- Piecewise / Hybrid Cliff Detection ---
    "piecewise_min_segment_laps": 4,
    "piecewise_min_slope_increase": 0.08,
    "piecewise_min_improvement_ratio": 0.20,

    # --- Strategy Useful Life ---
    "pit_loss_sec": 22.0,             # Fallback pit loss if track-specific unavailable
    "fuel_burn_sec_per_lap": 0.07,    # Fuel mass effect on lap time
    "sc_aging_reduction": 0.70,       # Safety-car laps count as 70% of normal wear
    "min_stint_laps": 8,              # Minimum viable stint length
    "max_stint_laps": 50,             # Hard ceiling for any stint
    "useful_life_uncertainty_draws": 200,
    "useful_life_uncertainty_seed": 42,
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
            "cliff_method": "sustained",
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
                    "cliff_method": "sustained",
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
        "cliff_method": "sustained",
        "cliff_reason": "no clear performance cliff detected",
    }


# --------------------------------------------------------------------------- #
#  2B. ROLLING-TREND SUSTAINED CLIFF DETECTION                                 #
# --------------------------------------------------------------------------- #
def _linear_trend(values: np.ndarray) -> tuple[float, float]:
    """Return the fitted slope and residual sum of squares for one window."""
    arr = np.asarray(values, dtype=float)
    x = np.arange(len(arr), dtype=float)
    x_centered = x - float(np.mean(x))
    y_mean = float(np.mean(arr))
    denominator = float(np.dot(x_centered, x_centered))
    slope = (
        0.0
        if denominator <= 1e-12
        else float(np.dot(x_centered, arr - y_mean) / denominator)
    )
    intercept = y_mean - slope * float(np.mean(x))
    residuals = arr - (intercept + slope * x)
    return slope, float(np.dot(residuals, residuals))


def _rolling_sustained_candidate(
    arr: np.ndarray,
    breakpoint: int,
    cfg: dict,
    baseline: float,
) -> dict:
    """Measure the local trend change around one zero-indexed breakpoint."""
    window = int(cfg["rolling_trend_window"])
    pre = arr[breakpoint - window:breakpoint]
    post = arr[breakpoint:breakpoint + window]
    local = arr[breakpoint - window:breakpoint + window]

    pre_slope, pre_sse = _linear_trend(pre)
    post_slope, post_sse = _linear_trend(post)
    _, single_sse = _linear_trend(local)
    split_sse = pre_sse + post_sse
    fit_improvement = (
        0.0
        if single_sse <= 1e-12
        else max(0.0, min(1.0, 1.0 - split_sse / single_sse))
    )
    slope_increase = post_slope - pre_slope
    post_step_median = float(np.median(np.diff(post)))
    supported_delta = float(post[-1] - baseline)

    qualifies = (
        post_slope >= float(cfg["cliff_slope_threshold"])
        and post_step_median >= float(cfg["cliff_slope_threshold"])
        and slope_increase >= float(cfg["rolling_min_slope_increase"])
        and supported_delta >= float(cfg["cliff_baseline_delta"])
        and fit_improvement
        >= float(cfg["rolling_min_fit_improvement_ratio"])
    )
    return {
        "breakpoint": breakpoint,
        "pre_slope": pre_slope,
        "post_slope": post_slope,
        "post_step_median": post_step_median,
        "slope_increase": slope_increase,
        "supported_delta": supported_delta,
        "fit_improvement_ratio": fit_improvement,
        "qualifies": qualifies,
    }


def detect_rolling_sustained_performance_cliff(
    smoothed_times: np.ndarray,
    config: dict = None,
) -> dict:
    """Detect a persistent local increase in the tire degradation trend.

    Unlike the legacy sustained detector, this candidate does not threshold
    noisy pointwise curvature. It compares short linear trends before and
    after each possible cliff, requires the split to improve the local fit,
    and confirms the signal at consecutive candidate laps.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(smoothed_times, dtype=float)
    n = len(arr)
    window = int(cfg["rolling_trend_window"])
    persistence = int(cfg["cliff_persistence_laps"])
    min_lap = int(cfg["cliff_min_lap"])

    if window < 2:
        raise ValueError("rolling_trend_window must be at least 2")
    if persistence < 1:
        raise ValueError("cliff_persistence_laps must be at least 1")

    first_breakpoint = max(window, min_lap - 1)
    last_breakpoint = n - window - persistence + 1
    if last_breakpoint < first_breakpoint:
        return {
            "performance_cliff_lap": None,
            "cliff_confidence": None,
            "cliff_method": "rolling_sustained",
            "cliff_reason": (
                "insufficient data for rolling pre- and post-cliff trends"
            ),
        }

    baseline_window = min(int(cfg["cliff_baseline_window"]), n)
    baseline = float(np.mean(arr[:baseline_window]))
    qualifying_run = []

    for breakpoint in range(first_breakpoint, last_breakpoint + 1):
        candidate = _rolling_sustained_candidate(
            arr,
            breakpoint,
            cfg,
            baseline,
        )
        if candidate["qualifies"]:
            qualifying_run.append(candidate)
        else:
            qualifying_run = []

        if len(qualifying_run) < persistence:
            continue

        first = qualifying_run[-persistence]
        strong_slope_change = (
            first["slope_increase"]
            >= 2 * float(cfg["rolling_min_slope_increase"])
        )
        strong_fit_improvement = (
            first["fit_improvement_ratio"]
            >= min(
                1.0,
                2 * float(cfg["rolling_min_fit_improvement_ratio"]),
            )
        )
        confidence = (
            "high"
            if strong_slope_change and strong_fit_improvement
            else "medium"
        )
        return {
            "performance_cliff_lap": int(first["breakpoint"] + 1),
            "cliff_confidence": confidence,
            "cliff_method": "rolling_sustained",
            "cliff_reason": (
                "persistent rolling trend increase detected: "
                f"{first['pre_slope']:.3f}s/lap to "
                f"{first['post_slope']:.3f}s/lap; "
                f"median supported step "
                f"{first['post_step_median']:.3f}s; "
                f"slope increase {first['slope_increase']:.3f}s/lap; "
                f"local fit improvement "
                f"{first['fit_improvement_ratio']:.1%}; "
                f"{persistence} consecutive candidate laps"
            ),
        }

    return {
        "performance_cliff_lap": None,
        "cliff_confidence": None,
        "cliff_method": "rolling_sustained",
        "cliff_reason": "no persistent rolling degradation increase detected",
    }


# --------------------------------------------------------------------------- #
#  2C. PIECEWISE-LINEAR CLIFF DETECTION                                        #
# --------------------------------------------------------------------------- #
def detect_piecewise_performance_cliff(
    smoothed_times: np.ndarray,
    config: dict = None,
) -> dict:
    """Detect a cliff as a statistically useful change in degradation slope."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(smoothed_times, dtype=float)
    n = len(arr)
    min_segment = int(cfg["piecewise_min_segment_laps"])

    if n < max(10, min_segment * 2):
        return {
            "performance_cliff_lap": None,
            "cliff_confidence": None,
            "cliff_method": "piecewise",
            "cliff_reason": "insufficient data for two piecewise segments",
        }

    x = np.arange(n, dtype=float)
    single_fit = np.polyfit(x, arr, 1)
    single_residuals = arr - np.polyval(single_fit, x)
    single_sse = float(np.sum(single_residuals ** 2))

    best = None
    for breakpoint in range(min_segment, n - min_segment + 1):
        pre_x = x[:breakpoint]
        post_x = x[breakpoint:]
        pre_fit = np.polyfit(pre_x, arr[:breakpoint], 1)
        post_fit = np.polyfit(post_x, arr[breakpoint:], 1)
        pre_slope = float(pre_fit[0])
        post_slope = float(post_fit[0])
        slope_increase = post_slope - pre_slope

        if post_slope < cfg["cliff_slope_threshold"]:
            continue
        if slope_increase < cfg["piecewise_min_slope_increase"]:
            continue

        pre_sse = float(
            np.sum((arr[:breakpoint] - np.polyval(pre_fit, pre_x)) ** 2)
        )
        post_sse = float(
            np.sum((arr[breakpoint:] - np.polyval(post_fit, post_x)) ** 2)
        )
        combined_sse = pre_sse + post_sse
        if best is None or combined_sse < best["combined_sse"]:
            best = {
                "breakpoint": breakpoint,
                "pre_slope": pre_slope,
                "post_slope": post_slope,
                "slope_increase": slope_increase,
                "combined_sse": combined_sse,
            }

    if best is None:
        return {
            "performance_cliff_lap": None,
            "cliff_confidence": None,
            "cliff_method": "piecewise",
            "cliff_reason": "no material piecewise slope increase detected",
        }

    if single_sse <= 1e-12:
        improvement_ratio = 0.0
    else:
        improvement_ratio = max(
            0.0, min(1.0, 1.0 - best["combined_sse"] / single_sse)
        )
    minimum_improvement = cfg["piecewise_min_improvement_ratio"]
    if improvement_ratio < minimum_improvement:
        return {
            "performance_cliff_lap": None,
            "cliff_confidence": None,
            "cliff_method": "piecewise",
            "cliff_reason": (
                "piecewise fit did not improve enough over a single degradation trend"
            ),
        }

    strong_slope_change = (
        best["slope_increase"] >= 2 * cfg["piecewise_min_slope_increase"]
    )
    strong_fit_improvement = improvement_ratio >= min(
        1.0, 2 * minimum_improvement
    )
    confidence = (
        "high" if strong_slope_change and strong_fit_improvement else "medium"
    )
    return {
        "performance_cliff_lap": int(best["breakpoint"] + 1),
        "cliff_confidence": confidence,
        "cliff_method": "piecewise",
        "cliff_reason": (
            "piecewise slope change detected: "
            f"{best['pre_slope']:.3f}s/lap to "
            f"{best['post_slope']:.3f}s/lap; "
            f"fit improvement {improvement_ratio:.1%}"
        ),
    }


def detect_hybrid_performance_cliff(
    smoothed_times: np.ndarray,
    config: dict = None,
) -> dict:
    """Confirm a piecewise breakpoint with sustained post-break degradation."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    piecewise = detect_piecewise_performance_cliff(smoothed_times, cfg)
    cliff_lap = piecewise["performance_cliff_lap"]
    if cliff_lap is None:
        return {
            **piecewise,
            "cliff_method": "hybrid",
            "cliff_reason": f"hybrid rejected: {piecewise['cliff_reason']}",
        }

    arr = np.asarray(smoothed_times, dtype=float)
    slope = np.gradient(arr)
    baseline_window = min(cfg["cliff_baseline_window"], len(arr))
    baseline = float(np.mean(arr[:baseline_window]))
    start_index = cliff_lap - 1
    persistence = int(cfg["cliff_persistence_laps"])
    end_index = start_index + persistence

    if end_index > len(arr):
        confirmed = False
    else:
        confirmed = all(
            slope[index] > cfg["cliff_slope_threshold"]
            and arr[index] - baseline > cfg["cliff_baseline_delta"]
            for index in range(start_index, end_index)
        )

    if not confirmed:
        return {
            "performance_cliff_lap": None,
            "cliff_confidence": None,
            "cliff_method": "hybrid",
            "cliff_reason": (
                "hybrid rejected: piecewise breakpoint lacked sustained "
                "post-break degradation"
            ),
        }

    return {
        **piecewise,
        "cliff_method": "hybrid",
        "cliff_reason": (
            f"hybrid confirmed at lap {cliff_lap}: {piecewise['cliff_reason']}"
        ),
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
    fuel_correction: bool = True,
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
    fuel_correction : bool
        Add back the fuel-mass benefit when raw observed lap times contain it.
        Set False for model curves generated at a fixed lap and fuel load.

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
    cum_cost = estimate_degradation_cost(
        arr,
        fuel_correction=fuel_correction,
        config=cfg,
    )

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


def estimate_useful_life_uncertainty(
    smoothed_times: np.ndarray,
    pit_loss: float = None,
    config: dict = None,
    fuel_correction: bool = True,
    residuals: np.ndarray | None = None,
    draws: int = None,
    seed: int = None,
) -> dict:
    """Estimate a bounded operational uncertainty range for useful life.

    The point estimate remains the deterministic pit-loss crossover.  Each
    perturbation resamples leakage-safe residuals when supplied (normally
    held-out residuals); otherwise it uses the local curve residuals as a
    deterministic inference fallback.  This is an operational range, not a
    formal confidence interval.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    arr = np.asarray(smoothed_times, dtype=float)
    point_result = recommend_strategy_useful_life(
        arr, pit_loss, cfg, fuel_correction=fuel_correction
    )
    point = int(point_result["strategy_useful_life_lap"])
    draws = int(draws or cfg["useful_life_uncertainty_draws"])
    seed = int(seed if seed is not None else cfg["useful_life_uncertainty_seed"])
    if draws < 1:
        raise ValueError("useful-life uncertainty draws must be positive")

    if residuals is None:
        # At inference there is no observed target. Estimate a small local
        # noise distribution from the curve itself rather than inventing a
        # compound-specific constant. Analysis runs should pass held-out
        # residuals explicitly.
        local_fit = np.polyval(np.polyfit(np.arange(len(arr)), arr, 1), np.arange(len(arr)))
        residual_pool = arr - local_fit
    else:
        residual_pool = np.asarray(residuals, dtype=float)
    residual_pool = residual_pool[np.isfinite(residual_pool)]
    if residual_pool.size == 0 or np.allclose(residual_pool, 0.0):
        residual_pool = np.asarray([0.0], dtype=float)

    rng = np.random.default_rng(seed)
    crossovers = []
    for _ in range(draws):
        perturbation = rng.choice(residual_pool, size=len(arr), replace=True)
        perturbed = arr + perturbation
        perturbed_result = recommend_strategy_useful_life(
            perturbed, pit_loss, cfg, fuel_correction=fuel_correction
        )
        crossovers.append(int(perturbed_result["strategy_useful_life_lap"]))

    empirical_lower = float(np.quantile(crossovers, 0.10))
    empirical_upper = float(np.quantile(crossovers, 0.90))
    raw_half_width = max(point - empirical_lower, empirical_upper - point)
    uncertainty_laps = int(min(3, max(1, np.ceil(raw_half_width))))
    capped = bool(raw_half_width > 3)
    lower = max(int(cfg["min_stint_laps"]), point - uncertainty_laps)
    upper = min(int(cfg["max_stint_laps"]), max(1, len(arr) - 1), point + uncertainty_laps)
    lower = min(lower, point)
    upper = max(upper, point)
    if point_result["strategy_confidence"] == "low" or capped:
        confidence = "low"
    elif uncertainty_laps == 1:
        confidence = "high"
    elif uncertainty_laps == 2:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "strategy_useful_life_lap": point,
        "strategy_useful_life_lower": int(lower),
        "strategy_useful_life_upper": int(upper),
        "strategy_useful_life_uncertainty_laps": uncertainty_laps,
        "strategy_useful_life_confidence": confidence,
        "strategy_useful_life_interval_method": "empirical_residual_perturbation",
        "strategy_useful_life_interval_capped": capped,
        "strategy_useful_life_empirical_lower": empirical_lower,
        "strategy_useful_life_empirical_upper": empirical_upper,
        "strategy_useful_life_crossover_samples": crossovers,
    }


# --------------------------------------------------------------------------- #
#  4. MASTER FUNCTION — combines both detectors                                #
# --------------------------------------------------------------------------- #
def analyze_tire_life(
    raw_times: list | np.ndarray,
    pit_loss: float = None,
    config: dict = None,
    fuel_correction: bool = True,
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
    fuel_correction : bool
        Whether to add a fuel-burn correction to strategic degradation cost.
        Model curves simulated at fixed fuel load must set this to False.

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
    cliff_method = cfg["cliff_detection_method"]
    detectors = {
        "sustained": detect_performance_cliff,
        "rolling_sustained": detect_rolling_sustained_performance_cliff,
        "piecewise": detect_piecewise_performance_cliff,
        "hybrid": detect_hybrid_performance_cliff,
    }
    if cliff_method not in detectors:
        raise ValueError(
            "cliff_detection_method must be sustained, rolling_sustained, "
            "piecewise, or hybrid"
        )
    cliff_result = detectors[cliff_method](smoothed, cfg)

    # ---- Step 3: Strategy useful life (race strategy) ----
    strategy_result = recommend_strategy_useful_life(
        smoothed,
        pit_loss,
        cfg,
        fuel_correction=fuel_correction,
    )
    uncertainty_result = estimate_useful_life_uncertainty(
        smoothed,
        pit_loss,
        cfg,
        fuel_correction=fuel_correction,
    )

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
        "cliff_method": cliff_result["cliff_method"],
        "cliff_reason": cliff_result["cliff_reason"],

        # Strategy useful life (race strategy)
        "strategy_useful_life_lap": strategy_result["strategy_useful_life_lap"],
        "strategy_confidence": uncertainty_result["strategy_useful_life_confidence"],
        "strategy_reason": strategy_result["strategy_reason"],
        "cumulative_deg_cost": strategy_result["cumulative_deg_cost"],
        "strategy_useful_life_lower": uncertainty_result["strategy_useful_life_lower"],
        "strategy_useful_life_upper": uncertainty_result["strategy_useful_life_upper"],
        "strategy_useful_life_uncertainty_laps": uncertainty_result["strategy_useful_life_uncertainty_laps"],
        "strategy_useful_life_interval_method": uncertainty_result["strategy_useful_life_interval_method"],
        "strategy_useful_life_interval_capped": uncertainty_result["strategy_useful_life_interval_capped"],
        "strategy_useful_life_empirical_lower": uncertainty_result["strategy_useful_life_empirical_lower"],
        "strategy_useful_life_empirical_upper": uncertainty_result["strategy_useful_life_empirical_upper"],
    }
