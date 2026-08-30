import itertools
import math

from mappings import TRACK_PIT_LOSS, TRACK_OVERTAKING_PENALTY

# Default Constants
FUEL_BURN_PER_LAP = 0.07
DIRTY_AIR_WEAR_MULTIPLIER = 1.05
SC_WEAR_MULTIPLIER = 0.70

# Strategy-mode constants
SC_PIT_LOSS_REDUCTION = 0.60  # Pit under SC costs ~60% of normal (40% saving)

# Hard absolute caps per compound — no strategy mode should ever exceed these.
# These exist as a safety net even when the degradation engine doesn't detect a cliff.
COMPOUND_ABSOLUTE_MAX_LAPS = {
    "SOFT": 25,
    "MEDIUM": 40,
    "HARD": 50,
    "INTERMEDIATE": 35,
    "WET": 40,
}

# Wet-weather scoring penalties/bonuses (seconds added to total delta)
SAFE_DRY_IN_WET_PENALTY = 15.0       # Safe: penalize staying on drys in any wet condition
RISKY_DRY_IN_LIGHT_WET_BONUS = -5.0  # Risky: reward gambling on drys in light wet

# Wet and intermediate tyres are trained/predicted in wet context. When a race
# scenario is dry but the car is still on wet-weather tyres, apply a strategy
# scoring correction so the raw wet-weather curve is not treated as competitive.
WET_TIRE_DRY_BASE_PENALTY_SEC = {
    "INTERMEDIATE": 3.0,
    "WET": 6.0,
}
WET_TIRE_DRY_AGE_PENALTY_SEC = {
    "INTERMEDIATE": 0.12,
    "WET": 0.25,
}

# Dry tyres retain less grip as rain increases. Softer compounds lose the
# least lap time; hards lose the most because they are harder to keep warm.
DRY_TIRE_WET_BASE_PENALTY_SEC = {
    "SOFT": 0.75,
    "MEDIUM": 1.0,
    "HARD": 1.25,
}
DRY_TIRE_HEAVY_WET_MULTIPLIER = 3.0

# Minimum delta difference (%) for safe/risky to be shown — if it's within this
# threshold of optimal, it's not meaningfully different and we return null
MIN_STRATEGY_DIVERGENCE_FRAC = 0.003  # 0.3% of optimal delta


class StrategySimulator:
    def __init__(self, degradation_profiles: dict):
        """
        Takes the output dictionary from TireDegradationSimulator.simulate().
        The degradation engine is consumed as-is — no modifications to its output.
        """
        self.profiles = degradation_profiles["compounds"]
        self.context = degradation_profiles["input_context"]
        # Normalize graph keys once. The previous implementation rebuilt an
        # int-keyed graph for every simulated lap in every candidate strategy,
        # which made a normal 72-lap request exceed deployment timeouts.
        self._graphs = {}
        self._graph_max_ages = {}
        for compound, profile in self.profiles.items():
            graph = {int(k): float(v) for k, v in profile["graph_data"].items()}
            self._graphs[compound] = graph
            self._graph_max_ages[compound] = max(graph)

        # Candidate strategies repeatedly evaluate the same compound/age and
        # compound/stint combinations. Cache both levels for the lifetime of a
        # single request.
        self._deg_time_cache = {}
        self._stint_cache = {}
        
    def _get_deg_time(self, compound: str, effective_age: float):
        """Linearly interpolates degradation time from the ML curve based on effective tire age."""
        age = round(float(effective_age), 6)
        cache_key = (compound, age)
        if cache_key in self._deg_time_cache:
            return self._deg_time_cache[cache_key]

        graph = self._graphs[compound]
        max_age = self._graph_max_ages[compound]
        
        if age <= 1.0:
            result = graph[1]
        elif age >= max_age:
            # If we blow past the curve data, extrapolate aggressively using the cliff slope
            overshoot = age - max_age
            result = graph[max_age] + (
                overshoot * self.profiles[compound]["drop_off_per_lap_sec"] * 2.0
            )
        else:
            lower_age = math.floor(age)
            upper_age = math.ceil(age)

            if lower_age == upper_age:
                result = graph.get(lower_age, 0.0)
            else:
                # Interpolate
                lower_val = graph.get(lower_age, 0.0)
                upper_val = graph.get(upper_age, 0.0)
                fraction = age - lower_age
                result = lower_val + (upper_val - lower_val) * fraction

        self._deg_time_cache[cache_key] = result
        return result

    def _wrong_condition_penalty(self, compound: str, effective_age: float,
                                 weather_condition: str):
        """Returns per-lap tyre mismatch penalty for strategy scoring."""
        if weather_condition in ("light_wet", "heavy_wet") and compound in DRY_TIRE_WET_BASE_PENALTY_SEC:
            penalty = DRY_TIRE_WET_BASE_PENALTY_SEC[compound]
            if weather_condition == "heavy_wet":
                penalty *= DRY_TIRE_HEAVY_WET_MULTIPLIER
            return penalty

        if weather_condition != "dry" or compound not in WET_TIRE_DRY_BASE_PENALTY_SEC:
            return 0.0

        age = max(1.0, float(effective_age))
        base_penalty = WET_TIRE_DRY_BASE_PENALTY_SEC[compound]
        age_penalty = WET_TIRE_DRY_AGE_PENALTY_SEC[compound] * max(0.0, age - 1.0)
        return base_penalty + age_penalty

    def _eval_stint_components(self, compound: str, length: int, position: int,
                               start_age: float = 0.0, sc_laps: int = 0,
                               sc_currently_out: bool = False,
                               weather_condition: str = "dry"):
        """Evaluate a stint and return an auditable cost/overshoot breakdown."""
        cache_key = (
            compound,
            int(length),
            int(position),
            round(float(start_age), 6),
            int(sc_laps),
            bool(sc_currently_out),
            weather_condition,
        )
        if cache_key in self._stint_cache:
            return self._stint_cache[cache_key]

        effective_age = float(start_age)
        if sc_laps > 0:
            effective_age -= sc_laps * (1.0 - SC_WEAR_MULTIPLIER)
            effective_age = max(0.0, effective_age)

        effective_start_age = effective_age
        fresh_lap_time = self._graphs[compound][1]
        base_lap_time_cost = 0.0
        degradation_cost = 0.0
        weather_mismatch_cost = 0.0
        for lap_in_stint in range(length):
            wear_factor = DIRTY_AIR_WEAR_MULTIPLIER if position > 1 else 1.0
            if sc_currently_out and lap_in_stint < 3:
                wear_factor *= SC_WEAR_MULTIPLIER

            effective_age += wear_factor
            predicted_lap_time = self._get_deg_time(compound, effective_age)
            base_lap_time_cost += fresh_lap_time
            degradation_cost += predicted_lap_time - fresh_lap_time
            weather_mismatch_cost += self._wrong_condition_penalty(
                compound, effective_age, weather_condition
            )

        useful_life = self._get_useful_life(compound)
        performance_cliff = self._get_performance_cliff(compound)
        useful_life_overshoot = max(0.0, effective_age - useful_life)
        performance_cliff_overshoot = (
            None
            if performance_cliff is None
            else max(0.0, effective_age - performance_cliff)
        )
        expected_lap_time_cost = (
            base_lap_time_cost + degradation_cost + weather_mismatch_cost
        )
        result = {
            "compound": compound,
            "length_laps": int(length),
            "effective_start_age_laps": effective_start_age,
            "effective_end_age_laps": effective_age,
            "base_lap_time_cost_sec": base_lap_time_cost,
            "degradation_cost_sec": degradation_cost,
            "weather_mismatch_cost_sec": weather_mismatch_cost,
            "expected_lap_time_cost_sec": expected_lap_time_cost,
            "performance_cliff_lap": performance_cliff,
            "performance_cliff_confidence": self.profiles[compound].get(
                "cliff_confidence"
            ),
            "performance_cliff_overshoot_laps": performance_cliff_overshoot,
            "strategy_useful_life_lap": useful_life,
            "strategy_useful_life_confidence": self.profiles[compound].get(
                "strategy_confidence"
            ),
            "useful_life_overshoot_laps": useful_life_overshoot,
        }
        self._stint_cache[cache_key] = result
        return result

    def _eval_stint(self, compound: str, length: int, position: int,
                    start_age: float = 0.0, sc_laps: int = 0,
                    sc_currently_out: bool = False,
                    weather_condition: str = "dry"):
        """Backward-compatible compact stint result used by existing callers."""
        components = self._eval_stint_components(
            compound,
            length,
            position,
            start_age=start_age,
            sc_laps=sc_laps,
            sc_currently_out=sc_currently_out,
            weather_condition=weather_condition,
        )
        return (
            components["expected_lap_time_cost_sec"],
            components["useful_life_overshoot_laps"],
        )

    def _get_useful_life(self, compound: str):
        """Returns the strategy useful life lap for a compound from degradation engine output."""
        return self.profiles[compound].get("strategy_useful_life_lap",
                    self.profiles[compound].get("recommended_max_life",
                    self.profiles[compound]["cliff_point_lap"]))

    def _get_performance_cliff(self, compound: str):
        """Return the physical performance cliff, if the detector found one."""
        return self.profiles[compound].get("performance_cliff_lap")

    def _get_stint_cap(self, compound: str, mode: str = "optimal"):
        """
        Returns the maximum allowed stint length for a compound based on strategy mode.
        
        - safe:    at or below the useful life (no extending)
        - optimal: bounded only by absolute max
        - risky:   bounded only by absolute max (willing to push to the limit)
        """
        useful_life = self._get_useful_life(compound)
        absolute_max = COMPOUND_ABSOLUTE_MAX_LAPS.get(compound, 50)
        
        if mode == "safe":
            # Safe: never exceed useful life
            return min(useful_life, absolute_max)
        else:
            # Optimal and Risky: can push up to absolute physical limits
            return absolute_max

    def _eval_strategy(self, compounds: list, stint_lengths: list, position: int,
                       laps_to_complete: int, start_age: float = 0.0,
                       sc_laps_on_first_stint: int = 0,
                       sc_currently_out: bool = False,
                       weather_condition: str = "dry"):
        """Evaluates total race delta time for a specific combination of stints."""
        base_lap_time_cost = 0.0
        degradation_cost = 0.0
        weather_mismatch_cost = 0.0
        max_useful_life_overshoot = 0.0
        max_future_useful_life_overshoot = 0.0
        performance_cliff_overshoots = []
        future_performance_cliff_overshoots = []
        stint_diagnostics = []
        
        for idx, (compound, length) in enumerate(zip(compounds, stint_lengths)):
            stint = self._eval_stint_components(
                compound,
                length,
                position,
                start_age=start_age if idx == 0 else 0.0,
                sc_laps=sc_laps_on_first_stint if idx == 0 else 0,
                sc_currently_out=sc_currently_out if idx == 0 else False,
                weather_condition=weather_condition,
            )
            stint_diagnostics.append(stint)
            base_lap_time_cost += stint["base_lap_time_cost_sec"]
            degradation_cost += stint["degradation_cost_sec"]
            weather_mismatch_cost += stint["weather_mismatch_cost_sec"]

            useful_overshoot = stint["useful_life_overshoot_laps"]
            max_useful_life_overshoot = max(
                max_useful_life_overshoot, useful_overshoot
            )
            performance_overshoot = stint[
                "performance_cliff_overshoot_laps"
            ]
            if performance_overshoot is not None:
                performance_cliff_overshoots.append(performance_overshoot)
            # Track overshoot on future fresh-tire stints separately.
            if idx > 0:
                max_future_useful_life_overshoot = max(
                    max_future_useful_life_overshoot, useful_overshoot
                )
                if performance_overshoot is not None:
                    future_performance_cliff_overshoots.append(
                        performance_overshoot
                    )
                
        # Total optimization time = Deg + Pit losses + Overtaking/Traffic penalty
        num_stops = len(compounds) - 1
        track_name = self.context["track"]
        pit_loss_sec = TRACK_PIT_LOSS.get(track_name, 24.0)
        overtaking_penalty_per_extra_stop = TRACK_OVERTAKING_PENALTY.get(track_name, 0.0)
        
        # If SC is currently out, reduce pit loss (opportunistic pitting)
        if sc_currently_out and num_stops > 0:
            # First stop gets the SC discount, remaining stops are normal
            total_pit_loss = (pit_loss_sec * SC_PIT_LOSS_REDUCTION) + max(0, (num_stops - 1)) * pit_loss_sec
        else:
            total_pit_loss = num_stops * pit_loss_sec
            
        # Overtaking penalty applies to stops beyond the first one.
        # Since the first stop is mandatory anyway, it doesn't incur extra traffic risk.
        # But additional stops drop you back into traffic and require overtaking.
        traffic_penalty = max(0, num_stops - 1) * overtaking_penalty_per_extra_stop
        
        expected_lap_time_cost = (
            base_lap_time_cost + degradation_cost + weather_mismatch_cost
        )
        total_time_delta = (
            expected_lap_time_cost + total_pit_loss + traffic_penalty
        )
        # Risk scoring is intentionally disabled in this first instrumentation
        # phase. Keeping it separate prevents diagnostics from changing rankings.
        cliff_risk_cost = 0.0
        risk_adjusted_total = total_time_delta + cliff_risk_cost
        
        return {
            "compounds": compounds,
            "stints": stint_lengths,
            "stops": num_stops,
            "total_delta": total_time_delta,
            "base_lap_time_cost_sec": base_lap_time_cost,
            "degradation_cost_sec": degradation_cost,
            "weather_mismatch_cost_sec": weather_mismatch_cost,
            "pit_loss_cost_sec": total_pit_loss,
            "traffic_cost_sec": traffic_penalty,
            "cliff_risk_cost_sec": cliff_risk_cost,
            "expected_total_time_sec": total_time_delta,
            "risk_adjusted_total_time_sec": risk_adjusted_total,
            "stint_diagnostics": stint_diagnostics,
            "max_performance_cliff_overshoot_laps": (
                max(performance_cliff_overshoots)
                if performance_cliff_overshoots
                else None
            ),
            "max_future_performance_cliff_overshoot_laps": (
                max(future_performance_cliff_overshoots)
                if future_performance_cliff_overshoots
                else None
            ),
            "max_useful_life_overshoot_laps": max_useful_life_overshoot,
            "max_future_useful_life_overshoot_laps": (
                max_future_useful_life_overshoot
            ),
            # Backward-compatible aliases. Historically "cliff" referred to
            # strategic useful life in this strategy engine.
            "max_cliff_overshoot": max_useful_life_overshoot,
            "max_future_cliff_overshoot": max_future_useful_life_overshoot,
        }

    def _apply_weather_adjustments(self, strategy: dict, weather_condition: str, mode: str):
        """
        Returns an adjusted total_delta for sorting purposes based on weather condition and strategy mode.
        Does not mutate the original strategy dict.
        """
        adjusted_delta = strategy["total_delta"]
        compounds = strategy["compounds"]
        
        has_wet_tires = any(c in ("INTERMEDIATE", "WET") for c in compounds)
        has_dry_tires = any(c in ("SOFT", "MEDIUM", "HARD") for c in compounds)
        
        if mode == "safe":
            # Safe: penalize any dry tires in wet conditions
            if weather_condition in ("light_wet", "heavy_wet") and has_dry_tires and not has_wet_tires:
                adjusted_delta += SAFE_DRY_IN_WET_PENALTY
            elif weather_condition == "light_wet" and has_dry_tires:
                # Mixed strategy in light wet: partial penalty
                adjusted_delta += SAFE_DRY_IN_WET_PENALTY * 0.5
                
        elif mode == "risky":
            # Risky: reward staying on drys in light wet
            if weather_condition == "light_wet" and has_dry_tires and not has_wet_tires:
                adjusted_delta += RISKY_DRY_IN_LIGHT_WET_BONUS
                
        return adjusted_delta

    def _stint_respects_cap(self, compounds: list, stint_lengths: list, mode: str,
                            start_age: float = 0.0):
        """
        Checks whether every stint in the strategy respects the compound-specific
        stint cap for the given mode (safe / optimal / risky).
        
        The inherited first stint (start_age > 0) is exempt from cap checks because
        the driver can't undo laps already driven. Only future fresh tire stints
        are checked.
        """
        for idx, (compound, length) in enumerate(zip(compounds, stint_lengths)):
            # Skip the inherited tire — we can't change laps already driven
            if idx == 0 and start_age > 0:
                continue
            cap = self._get_stint_cap(compound, mode)
            if length > cap:
                return False
        return True

    def _satisfies_compound_rule(self, compounds_used: list, planned_compounds: list,
                                 compounds_used_count: int = 0):
        """
        F1 requires at least two tyre compound types unless an intermediate or
        wet tyre is used during the race. A same-compound strategy is therefore
        only legal when that compound is INTERMEDIATE or WET.
        """
        compounds_in_race = set(compounds_used) | set(planned_compounds)
        known_distinct_compounds = len(compounds_in_race)
        distinct_compounds_used = max(known_distinct_compounds, int(compounds_used_count or 0))
        has_wet_weather_tyre = any(c in ("INTERMEDIATE", "WET") for c in compounds_in_race)
        return has_wet_weather_tyre or distinct_compounds_used >= 2

    def _is_light_wet_compound_sequence_allowed(self, compounds: list):
        """
        Light rain normally searches dry tyres plus intermediates. Full wets are
        only allowed in controlled transition patterns so they do not explode the
        mixed-weather candidate space.
        """
        if "WET" not in compounds:
            return True

        phases = []
        for compound in compounds:
            phase = "DRY" if compound in ("SOFT", "MEDIUM", "HARD") else compound
            if not phases or phases[-1] != phase:
                phases.append(phase)

        allowed_phases = {
            ("INTERMEDIATE", "WET"),
            ("WET", "INTERMEDIATE"),
            ("INTERMEDIATE", "WET", "INTERMEDIATE"),
            ("WET", "INTERMEDIATE", "WET"),
            ("DRY", "INTERMEDIATE", "WET"),
            ("WET", "INTERMEDIATE", "DRY"),
            ("INTERMEDIATE", "WET", "DRY"),
            ("DRY", "INTERMEDIATE", "WET", "INTERMEDIATE"),
            ("INTERMEDIATE", "WET", "INTERMEDIATE", "DRY"),
            ("DRY", "INTERMEDIATE", "WET", "INTERMEDIATE", "DRY"),
        }
        return tuple(phases) in allowed_phases

    def generate_strategies(self, total_laps: int, current_lap: int = 0,
                            current_compound: str = None, laps_on_current_tire: int = 0,
                            sc_happened_on_tire: bool = False, sc_laps_on_tire: int = 0,
                            sc_currently_out: bool = False,
                            has_pitted: bool = False,
                            track_position: int = 1, grid_pos: int = 1,
                            weather_condition: str = "dry",
                            compounds_used: list = None,
                            compounds_used_count: int = 0,
                            include_candidate_diagnostics: bool = False):
        """
        Brute forces 1, 2, and 3 stop strategies to find the mathematical optimum.
        Categorizes them into Optimal, Safe, and Risky.
        
        Args:
            total_laps: Total race laps (e.g. 72)
            current_lap: Current lap number (0 = pre-race / start)
            current_compound: Tire compound currently on the car (e.g. "MEDIUM")
            laps_on_current_tire: Laps already driven on the current tire set
            sc_happened_on_tire: Whether a safety car occurred during the current stint
            sc_laps_on_tire: Number of laps under SC on the current tire (reduces effective wear)
            sc_currently_out: Is a safety car currently deployed
            has_pitted: Whether the driver has already made at least 1 pit stop
            track_position: Current track position (used for dirty air when current_lap > 0)
            grid_pos: Starting grid position (used for dirty air when current_lap == 0)
            weather_condition: "dry", "light_wet", or "heavy_wet"
            compounds_used: List of compounds already used in the race (for 2-compound rule)
            compounds_used_count: Number of distinct compounds used so far, including current tire
        """
        # Derive laps remaining
        laps_remaining = total_laps - current_lap
        if laps_remaining <= 0:
            laps_remaining = 1
            
        # Determine position for dirty air calculation
        position = track_position if current_lap > 0 else grid_pos
        
        # Starting tire age for the first stint
        start_age = float(laps_on_current_tire) if current_compound else 0.0
        
        # SC laps on first stint
        sc_laps_first = sc_laps_on_tire if sc_happened_on_tire else 0
        
        # Track which compounds have already been used (for 2-compound rule)
        if compounds_used is None:
            compounds_used = []
        if current_compound and current_compound not in compounds_used:
            compounds_used.append(current_compound)
        compounds_used_count = max(int(compounds_used_count or 0), len(set(compounds_used)))
            
        # Determine available compounds based on weather
        dry_compounds = ["SOFT", "MEDIUM", "HARD"]
        wet_compounds = ["INTERMEDIATE", "WET"]
        
        if weather_condition == "heavy_wet":
            valid_compounds = wet_compounds
        elif weather_condition == "light_wet":
            valid_compounds = dry_compounds + wet_compounds
        else:
            valid_compounds = dry_compounds
        
        all_evaluations = []
        
        # Helper to generate stint length combinations that sum to laps_remaining.
        # Each stint is capped at the compound's absolute max to prune impossible combos.
        def get_stint_combos(compounds_seq, total_laps, step=2):
            num_stints = len(compounds_seq)
            # Per-stint absolute max (never exceed regardless of mode)
            maxes = [COMPOUND_ABSOLUTE_MAX_LAPS.get(c, 50) for c in compounds_seq]
            if num_stints >= 1 and current_compound:
                maxes[0] = max(0, maxes[0] - int(start_age))
                
            # The first stint (already underway) can have a short remaining length (minimum 1 lap).
            # Subsequent stints must be at least 5 laps.
            first_min = 1 if current_lap > 0 else 5
            sub_min = 5
            
            if num_stints == 1:
                if total_laps <= maxes[0]:
                    yield [total_laps]
                return
            elif num_stints == 2:
                hi_0 = min(maxes[0], total_laps - sub_min)
                for i in range(first_min, hi_0 + 1, step):
                    remainder = total_laps - i
                    if sub_min <= remainder <= maxes[1]:
                        yield [i, remainder]
            elif num_stints == 3:
                hi_0 = min(maxes[0], total_laps - sub_min * 2)
                for i in range(first_min, hi_0 + 1, step):
                    hi_1 = min(maxes[1], total_laps - i - sub_min)
                    for j in range(sub_min, hi_1 + 1, step):
                        remainder = total_laps - i - j
                        if sub_min <= remainder <= maxes[2]:
                            yield [i, j, remainder]
            elif num_stints == 4:
                s = step if total_laps < 50 else step * 2
                hi_0 = min(maxes[0], total_laps - sub_min * 3)
                for i in range(first_min, hi_0 + 1, s):
                    hi_1 = min(maxes[1], total_laps - i - sub_min * 2)
                    for j in range(sub_min, hi_1 + 1, s):
                        hi_2 = min(maxes[2], total_laps - i - j - sub_min)
                        for k in range(sub_min, hi_2 + 1, s):
                            remainder = total_laps - i - j - k
                            if sub_min <= remainder <= maxes[3]:
                                yield [i, j, k, remainder]
            elif num_stints == 5:
                # 4 stops (only for wet/chaotic races)
                s = step * 2 if total_laps < 50 else step * 3
                hi_0 = min(maxes[0], total_laps - sub_min * 4)
                for i in range(first_min, hi_0 + 1, s):
                    hi_1 = min(maxes[1], total_laps - i - sub_min * 3)
                    for j in range(sub_min, hi_1 + 1, s):
                        hi_2 = min(maxes[2], total_laps - i - j - sub_min * 2)
                        for k in range(sub_min, hi_2 + 1, s):
                            hi_3 = min(maxes[3], total_laps - i - j - k - sub_min)
                            for l in range(sub_min, hi_3 + 1, s):
                                remainder = total_laps - i - j - k - l
                                if sub_min <= remainder <= maxes[4]:
                                    yield [i, j, k, l, remainder]
        
        # Determine stop range
        # If hasn't pitted yet, must pit at least once (F1 rule)
        min_stops = 0 if has_pitted else 1
        
        # F1 rarely sees 4 stops unless conditions are chaotic (wet, SCs, mid-race pivots).
        # We cap pre-race dry strategies at 3 stops maximum to avoid computing impossible edge cases.
        if current_lap == 0 and weather_condition == "dry":
            max_stops = 3
        else:
            max_stops = 4
        
        for stops in range(min_stops, max_stops + 1):
            if stops == 0:
                # Zero-stop: stay out on current tire for remaining laps
                if not current_compound:
                    continue
                if not self._satisfies_compound_rule(
                    compounds_used,
                    [current_compound],
                    compounds_used_count=compounds_used_count,
                ):
                    continue
                # Don't allow zero-stop if remaining laps would blow past absolute max
                total_life = start_age + laps_remaining
                if total_life > COMPOUND_ABSOLUTE_MAX_LAPS.get(current_compound, 50):
                    continue
                    
                eval_result = self._eval_strategy(
                    [current_compound], [laps_remaining], position, laps_remaining,
                    start_age=start_age, sc_laps_on_first_stint=sc_laps_first,
                    sc_currently_out=sc_currently_out,
                    weather_condition=weather_condition,
                )
                all_evaluations.append(eval_result)
                continue
                
            num_stints = stops + 1
            
            # Generate compound permutations
            if current_compound:
                # First stint is forced to the current compound
                compound_pools = [[current_compound]] + [valid_compounds] * stops
                compound_combos = list(itertools.product(*compound_pools))
            else:
                compound_combos = list(itertools.product(valid_compounds, repeat=num_stints))
                
            for combo in compound_combos:
                if (
                    weather_condition == "light_wet"
                    and not self._is_light_wet_compound_sequence_allowed(combo)
                ):
                    continue
                if not self._satisfies_compound_rule(
                    compounds_used,
                    combo,
                    compounds_used_count=compounds_used_count,
                ):
                    continue

                for lengths in get_stint_combos(list(combo), laps_remaining):
                    eval_result = self._eval_strategy(
                        list(combo), lengths, position, laps_remaining,
                        start_age=start_age, sc_laps_on_first_stint=sc_laps_first,
                        sc_currently_out=sc_currently_out,
                        weather_condition=weather_condition,
                    )
                    all_evaluations.append(eval_result)
                    
        if not all_evaluations:
            return {
                "best_strategy": None,
                "safe_strategy": None,
                "risky_strategy": None
            }
            
        # Sort all by total race time delta (fastest first)
        all_evaluations.sort(key=lambda x: x["total_delta"])
        for rank, strategy in enumerate(all_evaluations, start=1):
            strategy["score_rank"] = rank
        
        # === OPTIMAL STRATEGY ===
        # Pure mathematical optimum — fastest total time
        best_strategy = all_evaluations[0]
        best_strategy["selection_basis"] = "lowest_expected_total_time"
        optimal_delta = best_strategy["total_delta"]
        optimal_stops = best_strategy["stops"]
        
        # === SAFE STRATEGY ===
        # Requirements:
        #   1. Every stint respects compound-specific useful life cap (no extending)
        #   2. Prefers more stops than optimal (splitting risk across shorter stints)
        #   3. Must be meaningfully different from optimal
        safe_candidates = [
            s for s in all_evaluations
            if self._stint_respects_cap(s["compounds"], s["stints"], "safe", start_age)
            and s["stops"] <= 2
        ]
        safe_selection_basis = "useful_life_constrained"
        
        # Fallback: If no perfectly safe 2-stop exists, find the 2-stops with the least cliff overshoot
        if not safe_candidates:
            safe_candidates = [s for s in all_evaluations if s["stops"] <= 2]
            safe_selection_basis = "minimum_useful_life_overshoot_fallback"
            # Sort by cliff overshoot first, then by total time
            safe_candidates.sort(key=lambda s: (s["max_cliff_overshoot"], s["total_delta"]))

        # Apply weather adjustments and re-rank
        if safe_candidates:
            safe_candidates.sort(key=lambda s: (s["max_cliff_overshoot"], self._apply_weather_adjustments(s, weather_condition, "safe")))
            safe_strategy = safe_candidates[0]
            safe_strategy["selection_basis"] = safe_selection_basis
        else:
            safe_strategy = None
            
        # === RISKY STRATEGY ===
        # Requirements:
        #   1. Same or FEWER stops than optimal (the risk is extending stints, not adding stops)
        #   2. Stints can exceed useful life up to the risky cap (1.5x useful life)
        #   3. At least one stint must be LONGER than optimal's longest (actually pushing limits)
        #   4. Must be meaningfully different from optimal
        optimal_max_stint = max(best_strategy["stints"])
        risky_candidates = [
            s for s in all_evaluations
            if s["stops"] <= optimal_stops
            and self._stint_respects_cap(s["compounds"], s["stints"], "risky", start_age)
            and max(s["stints"]) > optimal_max_stint
        ]
        # If no strategies with longer stints, fall back to same-stop-count strategies
        # that use different (harder) compounds — also a form of risk (longer on harder tire)
        if not risky_candidates:
            risky_candidates = [
                s for s in all_evaluations
                if s["stops"] < optimal_stops
                and self._stint_respects_cap(s["compounds"], s["stints"], "risky", start_age)
            ]
        # Apply weather adjustments and re-rank
        if risky_candidates:
            risky_candidates.sort(key=lambda s: self._apply_weather_adjustments(s, weather_condition, "risky"))
            risky_strategy = risky_candidates[0]
            risky_strategy["selection_basis"] = "extended_stint_candidate"
        else:
            risky_strategy = None
            
        # Ensure safe is different from optimal — if identical sequence, try to find a different sequence
        if safe_strategy:
            if safe_strategy["compounds"] == best_strategy["compounds"]:
                # Try to find a safe candidate with a DIFFERENT compound sequence
                for candidate in safe_candidates[1:]:
                    if candidate["compounds"] != best_strategy["compounds"]:
                        safe_strategy = candidate
                        break
                else:
                    diff = abs(safe_strategy["total_delta"] - best_strategy["total_delta"])
                    threshold = best_strategy["total_delta"] * MIN_STRATEGY_DIVERGENCE_FRAC
                    if diff < threshold:
                        safe_strategy = None  # Too similar to optimal in both sequence and time
                    
        # Ensure risky is different from optimal
        if risky_strategy:
            if risky_strategy["compounds"] == best_strategy["compounds"]:
                # Try to find a risky candidate with a DIFFERENT compound sequence
                for candidate in risky_candidates[1:]:
                    if candidate["compounds"] != best_strategy["compounds"]:
                        risky_strategy = candidate
                        break
                else:
                    # If only identical sequences exist, check if it's too close in overall race time
                    diff = abs(risky_strategy["total_delta"] - best_strategy["total_delta"])
                    threshold = best_strategy["total_delta"] * MIN_STRATEGY_DIVERGENCE_FRAC
                    if diff < threshold:
                        risky_strategy = None  # Too similar to optimal in both sequence and time

        if safe_strategy:
            safe_strategy["selection_basis"] = safe_selection_basis
        if risky_strategy:
            risky_strategy["selection_basis"] = "extended_stint_candidate"

        output = {
            "best_strategy": self._format_output(
                best_strategy, current_lap, "mathematical_fastest"
            ),
            "safe_strategy": self._format_output(
                safe_strategy, current_lap, "safe"
            ),
            "risky_strategy": self._format_output(
                risky_strategy, current_lap, "risky"
            ),
        }
        if include_candidate_diagnostics:
            output["candidate_diagnostics"] = [
                self._format_output(candidate, current_lap, "candidate")
                for candidate in all_evaluations[:5]
            ]
        return output
        
    def _format_output(self, strat, start_lap=0, strategy_role="candidate"):
        if not strat:
            return None
            
        sequence_labels = []
        stints_data = []
        current_lap = start_lap + 1 if start_lap > 0 else 1
        for comp, length, diagnostics in zip(
            strat["compounds"],
            strat["stints"],
            strat["stint_diagnostics"],
        ):
            end_lap = current_lap + length - 1
            sequence_labels.append(f"{comp} [L{current_lap} - L{end_lap}]")
            stints_data.append({
                "compound": comp,
                "laps": length,
                "start": current_lap,
                "end": end_lap,
                "effective_start_age_laps": round(
                    diagnostics["effective_start_age_laps"], 2
                ),
                "effective_end_age_laps": round(
                    diagnostics["effective_end_age_laps"], 2
                ),
                "performance_cliff_lap": diagnostics["performance_cliff_lap"],
                "performance_cliff_confidence": diagnostics[
                    "performance_cliff_confidence"
                ],
                "performance_cliff_overshoot_laps": (
                    None
                    if diagnostics["performance_cliff_overshoot_laps"] is None
                    else round(
                        diagnostics["performance_cliff_overshoot_laps"], 2
                    )
                ),
                "strategy_useful_life_lap": diagnostics[
                    "strategy_useful_life_lap"
                ],
                "strategy_useful_life_confidence": diagnostics[
                    "strategy_useful_life_confidence"
                ],
                "useful_life_overshoot_laps": round(
                    diagnostics["useful_life_overshoot_laps"], 2
                ),
                "base_lap_time_cost_sec": round(
                    diagnostics["base_lap_time_cost_sec"], 3
                ),
                "degradation_cost_sec": round(
                    diagnostics["degradation_cost_sec"], 3
                ),
                "weather_mismatch_cost_sec": round(
                    diagnostics["weather_mismatch_cost_sec"], 3
                ),
                "expected_lap_time_cost_sec": round(
                    diagnostics["expected_lap_time_cost_sec"], 3
                ),
            })
            current_lap = end_lap + 1
            
        return {
            "stops": strat["stops"],
            "strategy_role": strategy_role,
            "selection_basis": strat.get("selection_basis"),
            "score_rank": strat.get("score_rank"),
            "sequence": " -> ".join(sequence_labels),
            "stints_data": stints_data,
            "total_optimal_delta": round(strat["total_delta"], 2),
            "expected_total_time_sec": round(
                strat["expected_total_time_sec"], 3
            ),
            "risk_adjusted_total_time_sec": round(
                strat["risk_adjusted_total_time_sec"], 3
            ),
            "cost_breakdown": {
                "base_lap_time_cost_sec": round(
                    strat["base_lap_time_cost_sec"], 3
                ),
                "degradation_cost_sec": round(
                    strat["degradation_cost_sec"], 3
                ),
                "weather_mismatch_cost_sec": round(
                    strat["weather_mismatch_cost_sec"], 3
                ),
                "pit_loss_cost_sec": round(strat["pit_loss_cost_sec"], 3),
                "traffic_cost_sec": round(strat["traffic_cost_sec"], 3),
                "cliff_risk_cost_sec": round(
                    strat["cliff_risk_cost_sec"], 3
                ),
            },
            "max_performance_cliff_overshoot_laps": (
                None
                if strat["max_performance_cliff_overshoot_laps"] is None
                else round(
                    strat["max_performance_cliff_overshoot_laps"], 2
                )
            ),
            "max_useful_life_overshoot_laps": round(
                strat["max_useful_life_overshoot_laps"], 2
            ),
            "risk_cliff_overshoot": round(strat["max_cliff_overshoot"], 1),
        }

if __name__ == "__main__":
    import json
    from degradation_engine import TireDegradationSimulator
    
    # Quick Test — Pre-race scenario
    print("=== PRE-RACE TEST ===")
    print("Initializing Degradation Engine...")
    deg_sim = TireDegradationSimulator(year=2026, force_jit_check=False) 
    
    print("Generating profiles for Zandvoort...")
    out = deg_sim.simulate("VER", "Red Bull Racing", "Circuit Zandvoort", "2026-08-30", "15:00")
    
    # Print useful life values for context
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        ul = out["compounds"][comp].get("strategy_useful_life_lap")
        print(f"  {comp} useful life: {ul} laps")
    
    sim = StrategySimulator(out)
    result = sim.generate_strategies(total_laps=72, grid_pos=3)
    
    print("\n--- Optimal ---")
    print(json.dumps(result["best_strategy"], indent=2))
    print("\n--- Safe ---")
    print(json.dumps(result["safe_strategy"], indent=2))
    print("\n--- Risky ---")
    print(json.dumps(result["risky_strategy"], indent=2))
    
    # Quick Test — Mid-race scenario
    print("\n\n=== MID-RACE TEST (Lap 25, on Medium, 20 laps old) ===")
    result_mid = sim.generate_strategies(
        total_laps=72,
        current_lap=25,
        current_compound="MEDIUM",
        laps_on_current_tire=20,
        has_pitted=True,
        track_position=5,
        compounds_used=["SOFT", "MEDIUM"]
    )
    
    print("\n--- Optimal ---")
    print(json.dumps(result_mid["best_strategy"], indent=2))
    print("\n--- Safe ---")
    print(json.dumps(result_mid["safe_strategy"], indent=2))
    print("\n--- Risky ---")
    print(json.dumps(result_mid["risky_strategy"], indent=2))
