# Mappings for F1 Tire Degradation Model

from copy import deepcopy

from track_characteristics import (
    TRACK_CHARACTERISTICS,
    TRACK_CHARACTERISTIC_SOURCES,
    TRACK_FEATURE_NAMES,
)

# Normalization for Constructor/Team Names
# keys: raw names from FastF1/Official sources
# values: canonical name for the model
TEAM_MAPPING = {
    # --- Racing Bulls / AlphaTauri ---
    "Toro Rosso": "Racing Bulls",
    "Scuderia Toro Rosso": "Racing Bulls",
    "AlphaTauri": "Racing Bulls",
    "Scuderia AlphaTauri": "Racing Bulls",
    "AlphaTauri Honda": "Racing Bulls",
    "RB": "Racing Bulls",
    "RB F1 Team": "Racing Bulls",
    "Visa Cash App RB": "Racing Bulls",
    "Visa Cash App RB F1 Team": "Racing Bulls",
    "Racing Bulls": "Racing Bulls",
    "Racing Bulls F1 Team": "Racing Bulls",

    # --- Red Bull Racing (separate from Racing Bulls) ---
    "Red Bull": "Red Bull Racing",
    "Red Bull Racing": "Red Bull Racing",
    "Red Bull Racing Honda": "Red Bull Racing",
    "Red Bull Racing Honda RBPT": "Red Bull Racing",
    "Oracle Red Bull Racing": "Red Bull Racing",

    # --- Mercedes ---
    "Mercedes": "Mercedes",
    "Mercedes-AMG Petronas": "Mercedes",
    "Mercedes-AMG PETRONAS F1 Team": "Mercedes",
    "Mercedes-AMG Petronas Formula One Team": "Mercedes",

    # --- Ferrari ---
    "Ferrari": "Ferrari",
    "Scuderia Ferrari": "Ferrari",
    "Scuderia Ferrari Mission Winnow": "Ferrari",
    "Scuderia Ferrari HP": "Ferrari",

    # --- McLaren ---
    "McLaren": "McLaren",
    "McLaren F1 Team": "McLaren",
    "McLaren Renault": "McLaren",
    "McLaren Mercedes": "McLaren",

    # --- Williams ---
    "Williams": "Williams",
    "Williams Racing": "Williams",
    "Williams Mercedes": "Williams",

    # --- Haas ---
    "Haas": "Haas",
    "Haas F1 Team": "Haas",
    "Haas Ferrari": "Haas",
    "MoneyGram Haas F1 Team": "Haas",

    # --- Aston Martin (includes Racing Point era) ---
    "Racing Point": "Aston Martin",
    "Racing Point BWT Mercedes": "Aston Martin",
    "Aston Martin": "Aston Martin",
    "Aston Martin F1 Team": "Aston Martin",
    "Aston Martin Aramco Cognizant Formula One Team": "Aston Martin",
    "Aston Martin Aramco F1 Team": "Aston Martin",

    # --- Alpine (includes Renault era) ---
    "Renault": "Alpine",
    "Renault F1 Team": "Alpine",
    "Alpine": "Alpine",
    "Alpine F1 Team": "Alpine",
    "BWT Alpine F1 Team": "Alpine",

    # --- Audi (includes Sauber/Alfa Romeo era) ---
    "Alfa Romeo": "Audi",
    "Alfa Romeo Racing": "Audi",
    "Alfa Romeo Racing Ferrari": "Audi",
    "Alfa Romeo F1 Team": "Audi",
    "Sauber": "Audi",
    "Sauber F1 Team": "Audi",
    "Kick Sauber": "Audi",
    "Stake F1 Team Kick Sauber": "Audi",
    "Audi": "Audi",
    "Audi F1 Team": "Audi",
    "Audi Revolut F1 Team": "Audi",

    # --- Cadillac (new in 2026) ---
    "Cadillac": "Cadillac",
    "Cadillac Racing": "Cadillac",
    "Cadillac F1 Team": "Cadillac",
}


# ============================================================
# TRACK CONFIG — type, length, and official race lap count
# ============================================================
TRACK_CONFIG = {
    # === SLOW TRACKS ===
    'Circuit de Monaco (Monaco)': {'type': 'Slow', 'length_km': 3.337, 'race_laps': 78},
    'Marina Bay Street Circuit (Singapore)': {'type': 'Slow', 'length_km': 4.940, 'race_laps': 62},
    'Hungaroring (Hungary)': {'type': 'Slow', 'length_km': 4.381, 'race_laps': 70},
    'Circuit Zandvoort (Netherlands)': {'type': 'Slow', 'length_km': 4.259, 'race_laps': 72},
    'Miami International Autodrome (Miami, USA)': {'type': 'Slow', 'length_km': 5.412, 'race_laps': 57},
    'Autódromo Hermanos Rodríguez (Mexico)': {'type': 'Slow', 'length_km': 4.304, 'race_laps': 71},
    'MADRING (Madrid, Spain)': {'type': 'Slow', 'length_km': 5.470, 'race_laps': 57},

    # === MEDIUM TRACKS ===
    'Bahrain International Circuit (Bahrain)': {'type': 'Medium', 'length_km': 5.412, 'race_laps': 57},
    'Albert Park Grand Prix Circuit (Australia)': {'type': 'Medium', 'length_km': 5.278, 'race_laps': 58},
    'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)': {'type': 'Medium', 'length_km': 4.909, 'race_laps': 63},
    'Circuit de Barcelona-Catalunya (Spain)': {'type': 'Medium', 'length_km': 4.657, 'race_laps': 66},
    'Circuit de Barcelona-Catalunya (Barcelona, Spain)': {'type': 'Medium', 'length_km': 4.657, 'race_laps': 66},
    'Baku City Circuit (Azerbaijan)': {'type': 'Medium', 'length_km': 6.003, 'race_laps': 51},
    'Circuit Gilles Villeneuve (Canada)': {'type': 'Medium', 'length_km': 4.361, 'race_laps': 70},
    'Red Bull Ring (Austria)': {'type': 'Medium', 'length_km': 4.318, 'race_laps': 71},
    'Circuit Paul Ricard (France)': {'type': 'Medium', 'length_km': 5.842, 'race_laps': 53},
    'Suzuka Circuit (Japan)': {'type': 'Medium', 'length_km': 5.807, 'race_laps': 53},
    'Circuit of The Americas (Austin, USA)': {'type': 'Medium', 'length_km': 5.513, 'race_laps': 56},
    'Autódromo José Carlos Pace (Brazil)': {'type': 'Medium', 'length_km': 4.309, 'race_laps': 71},
    'Yas Marina Circuit (UAE)': {'type': 'Medium', 'length_km': 5.281, 'race_laps': 58},
    'Shanghai International Circuit (China)': {'type': 'Medium', 'length_km': 5.451, 'race_laps': 56},
    'Lusail International Circuit (Qatar)': {'type': 'Medium', 'length_km': 5.419, 'race_laps': 57},

    # === FAST TRACKS ===
    'Jeddah Corniche Circuit (Saudi Arabia)': {'type': 'Fast', 'length_km': 6.174, 'race_laps': 50},
    'Silverstone Circuit (Great Britain)': {'type': 'Fast', 'length_km': 5.891, 'race_laps': 52},
    'Circuit de Spa-Francorchamps (Belgium)': {'type': 'Fast', 'length_km': 7.004, 'race_laps': 44},
    'Autodromo Nazionale Monza (Monza, Italy)': {'type': 'Fast', 'length_km': 5.793, 'race_laps': 53},
    'Las Vegas Strip Circuit (Las Vegas, USA)': {'type': 'Fast', 'length_km': 6.201, 'race_laps': 50},
    'Autodromo Internazionale del Mugello (Italy)': {'type': 'Fast', 'length_km': 5.245, 'race_laps': 59},
}


# ============================================================
# TRACK CHARACTERISTICS — source-backed normalized 0.0–1.0 priors
# Raw Pirelli ratings, Mercedes corner speeds, source URLs, and the
# deterministic derivation live in track_characteristics.py and data/.
# ============================================================
# Default characteristics for unknown tracks
_DEFAULT_CHARACTERISTICS = {feature: 0.50 for feature in TRACK_FEATURE_NAMES}

# ============================================================
# EVENT NAME → CIRCUIT KEY BRIDGE
# FastF1 uses EventName (e.g. "Bahrain Grand Prix") but our
# TRACK_CONFIG / TRACK_CHARACTERISTICS use official circuit names
# (e.g. "Bahrain International Circuit (Bahrain)").
# This mapping resolves the former to the latter so that the
# preprocessing pipeline (which sees EventName) gets real track
# characteristics instead of falling back to neutral 0.5 defaults.
# ============================================================
EVENT_NAME_TO_CIRCUIT = {
    # 2022-2026 calendar events
    'Bahrain Grand Prix':          'Bahrain International Circuit (Bahrain)',
    'Saudi Arabian Grand Prix':    'Jeddah Corniche Circuit (Saudi Arabia)',
    'Australian Grand Prix':       'Albert Park Grand Prix Circuit (Australia)',
    'Emilia Romagna Grand Prix':   'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)',
    'Miami Grand Prix':            'Miami International Autodrome (Miami, USA)',
    'Spanish Grand Prix':          'Circuit de Barcelona-Catalunya (Spain)',
    'Barcelona Grand Prix':        'Circuit de Barcelona-Catalunya (Barcelona, Spain)',
    'Monaco Grand Prix':           'Circuit de Monaco (Monaco)',
    'Azerbaijan Grand Prix':       'Baku City Circuit (Azerbaijan)',
    'Canadian Grand Prix':         'Circuit Gilles Villeneuve (Canada)',
    'British Grand Prix':          'Silverstone Circuit (Great Britain)',
    'Austrian Grand Prix':         'Red Bull Ring (Austria)',
    'French Grand Prix':           'Circuit Paul Ricard (France)',
    'Hungarian Grand Prix':        'Hungaroring (Hungary)',
    'Belgian Grand Prix':          'Circuit de Spa-Francorchamps (Belgium)',
    'Dutch Grand Prix':            'Circuit Zandvoort (Netherlands)',
    'Italian Grand Prix':          'Autodromo Nazionale Monza (Monza, Italy)',
    'Singapore Grand Prix':        'Marina Bay Street Circuit (Singapore)',
    'Japanese Grand Prix':         'Suzuka Circuit (Japan)',
    'Qatar Grand Prix':            'Lusail International Circuit (Qatar)',
    'United States Grand Prix':    'Circuit of The Americas (Austin, USA)',
    'Mexico City Grand Prix':      'Autódromo Hermanos Rodríguez (Mexico)',
    'São Paulo Grand Prix':        'Autódromo José Carlos Pace (Brazil)',
    'Las Vegas Grand Prix':        'Las Vegas Strip Circuit (Las Vegas, USA)',
    'Abu Dhabi Grand Prix':        'Yas Marina Circuit (UAE)',
    'Chinese Grand Prix':          'Shanghai International Circuit (China)',
    'Madrid Grand Prix':           'MADRING (Madrid, Spain)',
    # Mugello (historical / possible future use)
    'Tuscan Grand Prix':           'Autodromo Internazionale del Mugello (Italy)',
}


def resolve_circuit_key(name):
    """
    Resolves any track identifier (FastF1 EventName, old circuit key, or new
    circuit key with country) to the canonical TRACK_CONFIG key.
    Returns the input unchanged if no mapping is found.
    """
    # 1. Direct match against TRACK_CONFIG (UI / inference path)
    if name in TRACK_CONFIG:
        return name
    # 2. FastF1 EventName → circuit key (training path)
    if name in EVENT_NAME_TO_CIRCUIT:
        return EVENT_NAME_TO_CIRCUIT[name]
    # 3. No match — return as-is (will fall back to defaults downstream)
    return name


def get_track_features(track_name):
    """
    Return the seven source-backed normalized features for a circuit.

    Circuits without an official catalogue row use the documented neutral
    fallback rather than hand-estimated values.
    """
    key = resolve_circuit_key(track_name)
    return TRACK_CHARACTERISTICS.get(key, _DEFAULT_CHARACTERISTICS).copy()


def get_track_characteristic_source(track_name):
    """Return source metadata for a circuit, or ``None`` for neutral fallback."""
    key = resolve_circuit_key(track_name)
    source = TRACK_CHARACTERISTIC_SOURCES.get(key)
    return deepcopy(source) if source else None


def get_legacy_track_feature_aliases(features):
    """Project sourced features onto the schema used by committed old models.

    This exists only so a deployment remains operational until its model is
    retrained.  New training runs use the source-native names directly and do
    not include the former uncited weighted composites.
    """
    return {
        'high_speed_load': features['corner_speed_energy'],
        'surface_roughness': features['asphalt_grip'],
        'track_temp_sensitivity': features['tyre_stress'],
        'thermal_stress': features['tyre_stress'],
        'surface_wear': features['abrasiveness'],
        'energy_load': features['corner_speed_energy'],
    }


def get_track_info(circuit_name):
    """
    Returns track classification, length, and race lap count for a given circuit name.
    """
    key = resolve_circuit_key(circuit_name)
    if key in TRACK_CONFIG:
        return TRACK_CONFIG[key]
    # Fallback defaults
    return {'type': 'Medium', 'length_km': 5.0, 'race_laps': 57}

def normalize_team_name(team_name):
    """
    Normalizes team name to a canonical ID.
    """
    return TEAM_MAPPING.get(team_name, team_name)

TRACK_PIT_LOSS = {
    'Bahrain International Circuit (Bahrain)': 21.0,
    'Jeddah Corniche Circuit (Saudi Arabia)': 20.0,
    'Albert Park Grand Prix Circuit (Australia)': 21.0,
    'Baku City Circuit (Azerbaijan)': 21.0,
    'Miami International Autodrome (Miami, USA)': 20.0,
    'Circuit de Monaco (Monaco)': 19.0,
    'Circuit de Barcelona-Catalunya (Spain)': 22.0,
    'Circuit de Barcelona-Catalunya (Barcelona, Spain)': 22.0,
    'Circuit Gilles Villeneuve (Canada)': 20.0,
    'Red Bull Ring (Austria)': 20.0,
    'Silverstone Circuit (Great Britain)': 24.0,
    'Hungaroring (Hungary)': 22.0,
    'Circuit de Spa-Francorchamps (Belgium)': 23.0,
    'Circuit Zandvoort (Netherlands)': 20.0,
    'Autodromo Nazionale Monza (Monza, Italy)': 24.0,
    'Marina Bay Street Circuit (Singapore)': 26.0,
    'Suzuka Circuit (Japan)': 22.0,
    'Lusail International Circuit (Qatar)': 21.0,
    'Circuit of The Americas (Austin, USA)': 22.0,
    'Autódromo Hermanos Rodríguez (Mexico)': 22.0,
    'Autódromo José Carlos Pace (Brazil)': 21.0,
    'Las Vegas Strip Circuit (Las Vegas, USA)': 20.0,
    'Yas Marina Circuit (UAE)': 22.0,
    'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)': 25.0,
    'Shanghai International Circuit (China)': 23.0,
    'Autodromo Internazionale del Mugello (Italy)': 25.0,
    'Circuit Paul Ricard (France)': 25.0,
    'MADRING (Madrid, Spain)': 23.0
}

# Penalty (in seconds) applied to each stop BEYOND the first one.
# This represents the virtual track position/traffic penalty on circuits where overtaking is difficult.
TRACK_OVERTAKING_PENALTY = {
    'Circuit de Monaco (Monaco)': 15.0,         # Overtaking is almost impossible, 2-stop is highly penalized
    'Marina Bay Street Circuit (Singapore)': 8.0, # Street circuit, very hard to pass
    'Hungaroring (Hungary)': 6.0,                # "Monaco without barriers", very hard to pass
    'Circuit Zandvoort (Netherlands)': 5.0,      # Narrow, twisty track
    'Circuit de Barcelona-Catalunya (Spain)': 4.0,
    'Circuit de Barcelona-Catalunya (Barcelona, Spain)': 4.0,
    'Autódromo Hermanos Rodríguez (Mexico)': 3.0,
    'Albert Park Grand Prix Circuit (Australia)': 3.0,
    'Yas Marina Circuit (UAE)': 2.0,
    'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)': 3.0,
    # High-overtaking tracks get 0.0 penalty
    'Silverstone Circuit (Great Britain)': 0.0,
    'Circuit de Spa-Francorchamps (Belgium)': 0.0,
    'Autodromo Nazionale Monza (Monza, Italy)': 0.0,
    'Las Vegas Strip Circuit (Las Vegas, USA)': 0.0,
    'Bahrain International Circuit (Bahrain)': 0.0,
    'Jeddah Corniche Circuit (Saudi Arabia)': 0.0,
    'Baku City Circuit (Azerbaijan)': 0.0,
    'Circuit Gilles Villeneuve (Canada)': 0.0,
    'Red Bull Ring (Austria)': 0.0,
    'Suzuka Circuit (Japan)': 0.0,
    'Lusail International Circuit (Qatar)': 0.0,
    'Circuit of The Americas (Austin, USA)': 0.0,
    'Autódromo José Carlos Pace (Brazil)': 0.0,
    'Shanghai International Circuit (China)': 0.0,
    'Autodromo Internazionale del Mugello (Italy)': 0.0,
    'Circuit Paul Ricard (France)': 0.0,
    'MADRING (Madrid, Spain)': 2.0
}


# Approximate race pace (seconds) for absolute lap time reconstruction
TRACK_BASE_PACE = {
    'Bahrain International Circuit (Bahrain)': 96.0,
    'Jeddah Corniche Circuit (Saudi Arabia)': 93.0,
    'Albert Park Grand Prix Circuit (Australia)': 82.0,
    'Baku City Circuit (Azerbaijan)': 106.0,
    'Miami International Autodrome (Miami, USA)': 92.0,
    'Circuit de Monaco (Monaco)': 76.0,
    'Circuit de Barcelona-Catalunya (Spain)': 80.0,
    'Circuit de Barcelona-Catalunya (Barcelona, Spain)': 80.0,
    'Circuit Gilles Villeneuve (Canada)': 76.0,
    'Red Bull Ring (Austria)': 69.0,
    'Silverstone Circuit (Great Britain)': 91.0,
    'Hungaroring (Hungary)': 83.0,
    'Circuit de Spa-Francorchamps (Belgium)': 111.0,
    'Circuit Zandvoort (Netherlands)': 75.0,
    'Autodromo Nazionale Monza (Monza, Italy)': 85.0,
    'Marina Bay Street Circuit (Singapore)': 97.0,
    'Suzuka Circuit (Japan)': 95.0,
    'Lusail International Circuit (Qatar)': 87.0,
    'Circuit of The Americas (Austin, USA)': 99.0,
    'Autódromo Hermanos Rodríguez (Mexico)': 83.0,
    'Autódromo José Carlos Pace (Brazil)': 75.0,
    'Las Vegas Strip Circuit (Las Vegas, USA)': 96.0,
    'Yas Marina Circuit (UAE)': 89.0,
    'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)': 79.0,
    'Shanghai International Circuit (China)': 98.0,
    'Autodromo Internazionale del Mugello (Italy)': 81.0,
    'Circuit Paul Ricard (France)': 96.0,
    'MADRING (Madrid, Spain)': 86.0
}

# Base driver roster for 2026 onwards
BASE_ROSTER = {
    "Red Bull Racing": ["VER", "HAD"],
    "Ferrari": ["LEC", "HAM"],
    "Mercedes": ["RUS", "ANT"],
    "McLaren": ["NOR", "PIA"],
    "Alpine": ["GAS", "COL"],
    "Audi": ["HUL", "BOR"],
    "Aston Martin": ["ALO", "STR"],
    "Haas": ["BEA", "OCO"],
    "Racing Bulls": ["LAW", "LIN"],
    "Williams": ["ALB", "SAI"],
    "Cadillac": ["BOT", "PER"],
}

# Overrides for specific years
ROSTER_OVERRIDES = {
    2022: {
        "Red Bull Racing": ["VER", "PER"],
        "Ferrari": ["LEC", "SAI"],
        "Mercedes": ["RUS", "HAM"],
        "McLaren": ["NOR", "RIC"],
        "Alpine": ["OCO", "ALO"],
        "Alfa Romeo": ["BOT", "ZHO"],
        "Aston Martin": ["VET", "STR", "HUL"],
        "Haas": ["MAG", "MSC"],
        "AlphaTauri": ["GAS", "TSU"],
        "Williams": ["ALB", "LAT", "DEV"],
    },
    2023: {
        "Red Bull Racing": ["VER", "PER"],
        "Ferrari": ["LEC", "SAI"],
        "Mercedes": ["RUS", "HAM"],
        "McLaren": ["NOR", "PIA"],
        "Alpine": ["GAS", "OCO"],
        "Alfa Romeo": ["BOT", "ZHO"],
        "Aston Martin": ["ALO", "STR"],
        "Haas": ["HUL", "MAG"],
        "AlphaTauri": ["TSU", "RIC", "LAW", "DEV"],
        "Williams": ["ALB", "SAR"],
    },
    2024: {
        "Red Bull Racing": ["VER", "PER"],
        "Ferrari": ["LEC", "SAI"],
        "Mercedes": ["RUS", "HAM"],
        "McLaren": ["NOR", "PIA"],
        "Alpine": ["GAS", "OCO", "DOO"],
        "Kick Sauber": ["BOT", "ZHO"],
        "Aston Martin": ["ALO", "STR"],
        "Haas": ["HUL", "MAG", "BEA"],
        "Racing Bulls": ["TSU", "RIC", "LAW"],
        "Williams": ["ALB", "SAR", "COL"],
    },
    2025: {
        "Red Bull Racing": ["VER", "TSU"],
        "Ferrari": ["LEC", "HAM"],
        "Mercedes": ["RUS", "ANT"],
        "McLaren": ["NOR", "PIA"],
        "Alpine": ["GAS", "DOO", "COL"],
        "Kick Sauber": ["HUL", "BOR"],
        "Aston Martin": ["ALO", "STR"],
        "Haas": ["BEA", "OCO"],
        "Racing Bulls": ["LAW", "HAD"],
        "Williams": ["ALB", "SAI"],
    },
    2026: {
        "Red Bull Racing": ["VER", "HAD"],
        "Ferrari": ["LEC", "HAM"],
        "Mercedes": ["RUS", "ANT"],
        "McLaren": ["NOR", "PIA"],
        "Alpine": ["GAS", "COL"],
        "Audi": ["HUL", "BOR"],
        "Aston Martin": ["ALO", "STR"],
        "Haas": ["BEA", "OCO"],
        "Racing Bulls": ["LAW", "LIN"],
        "Williams": ["ALB", "SAI"],
        "Cadillac": ["BOT", "PER"],
    }
}

def get_roster_map(years=None):
    """Generates a mapping of requested year -> team -> [drivers]."""
    full_map = {}

    requested_years = years if years is not None else range(2022, 2031)
    for year in requested_years:
        if year in ROSTER_OVERRIDES:
            full_map[year] = ROSTER_OVERRIDES[year]
        else:
            # Fallback to 2026 roster for future years
            full_map[year] = ROSTER_OVERRIDES[2026]
            
    return full_map

YEARLY_TRACKS = {
    2022: [
        'Bahrain International Circuit (Bahrain)', 'Jeddah Corniche Circuit (Saudi Arabia)', 'Albert Park Grand Prix Circuit (Australia)',
        'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)', 'Miami International Autodrome (Miami, USA)',
        'Circuit de Barcelona-Catalunya (Spain)', 'Circuit de Monaco (Monaco)', 'Baku City Circuit (Azerbaijan)',
        'Circuit Gilles Villeneuve (Canada)', 'Silverstone Circuit (Great Britain)', 'Red Bull Ring (Austria)',
        'Circuit Paul Ricard (France)', 'Hungaroring (Hungary)', 'Circuit de Spa-Francorchamps (Belgium)',
        'Circuit Zandvoort (Netherlands)', 'Autodromo Nazionale Monza (Monza, Italy)', 'Marina Bay Street Circuit (Singapore)',
        'Suzuka Circuit (Japan)', 'Circuit of The Americas (Austin, USA)', 'Autódromo Hermanos Rodríguez (Mexico)',
        'Autódromo José Carlos Pace (Brazil)', 'Yas Marina Circuit (UAE)'
    ],
    2023: [
        'Bahrain International Circuit (Bahrain)', 'Jeddah Corniche Circuit (Saudi Arabia)', 'Albert Park Grand Prix Circuit (Australia)',
        'Baku City Circuit (Azerbaijan)', 'Miami International Autodrome (Miami, USA)', 'Circuit de Monaco (Monaco)',
        'Circuit de Barcelona-Catalunya (Spain)', 'Circuit Gilles Villeneuve (Canada)', 'Red Bull Ring (Austria)',
        'Silverstone Circuit (Great Britain)', 'Hungaroring (Hungary)', 'Circuit de Spa-Francorchamps (Belgium)',
        'Circuit Zandvoort (Netherlands)', 'Autodromo Nazionale Monza (Monza, Italy)', 'Marina Bay Street Circuit (Singapore)',
        'Suzuka Circuit (Japan)', 'Lusail International Circuit (Qatar)', 'Circuit of The Americas (Austin, USA)',
        'Autódromo Hermanos Rodríguez (Mexico)', 'Autódromo José Carlos Pace (Brazil)', 'Las Vegas Strip Circuit (Las Vegas, USA)',
        'Yas Marina Circuit (UAE)'
    ],
    2024: [
        'Bahrain International Circuit (Bahrain)', 'Jeddah Corniche Circuit (Saudi Arabia)', 'Albert Park Grand Prix Circuit (Australia)',
        'Suzuka Circuit (Japan)', 'Shanghai International Circuit (China)', 'Miami International Autodrome (Miami, USA)',
        'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)', 'Circuit de Monaco (Monaco)', 'Circuit Gilles Villeneuve (Canada)',
        'Circuit de Barcelona-Catalunya (Spain)', 'Red Bull Ring (Austria)', 'Silverstone Circuit (Great Britain)',
        'Hungaroring (Hungary)', 'Circuit de Spa-Francorchamps (Belgium)', 'Circuit Zandvoort (Netherlands)',
        'Autodromo Nazionale Monza (Monza, Italy)', 'Baku City Circuit (Azerbaijan)', 'Marina Bay Street Circuit (Singapore)',
        'Circuit of The Americas (Austin, USA)', 'Autódromo Hermanos Rodríguez (Mexico)', 'Autódromo José Carlos Pace (Brazil)',
        'Las Vegas Strip Circuit (Las Vegas, USA)', 'Lusail International Circuit (Qatar)', 'Yas Marina Circuit (UAE)'
    ],
    2025: [
        'Albert Park Grand Prix Circuit (Australia)', 'Shanghai International Circuit (China)', 'Suzuka Circuit (Japan)',
        'Bahrain International Circuit (Bahrain)', 'Jeddah Corniche Circuit (Saudi Arabia)', 'Miami International Autodrome (Miami, USA)',
        'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)', 'Circuit de Monaco (Monaco)', 'Circuit de Barcelona-Catalunya (Spain)',
        'Circuit Gilles Villeneuve (Canada)', 'Red Bull Ring (Austria)', 'Silverstone Circuit (Great Britain)',
        'Circuit de Spa-Francorchamps (Belgium)', 'Hungaroring (Hungary)', 'Circuit Zandvoort (Netherlands)',
        'Autodromo Nazionale Monza (Monza, Italy)', 'Baku City Circuit (Azerbaijan)', 'Marina Bay Street Circuit (Singapore)',
        'Circuit of The Americas (Austin, USA)', 'Autódromo Hermanos Rodríguez (Mexico)', 'Autódromo José Carlos Pace (Brazil)',
        'Las Vegas Strip Circuit (Las Vegas, USA)', 'Lusail International Circuit (Qatar)', 'Yas Marina Circuit (UAE)'
    ],
    2026: [
        'Albert Park Grand Prix Circuit (Australia)', 'Shanghai International Circuit (China)', 'Suzuka Circuit (Japan)',
        'Miami International Autodrome (Miami, USA)', 'Circuit Gilles Villeneuve (Canada)', 'Circuit de Monaco (Monaco)',
        'Circuit de Barcelona-Catalunya (Barcelona, Spain)', 'Red Bull Ring (Austria)', 'Silverstone Circuit (Great Britain)',
        'Circuit de Spa-Francorchamps (Belgium)', 'Hungaroring (Hungary)', 'Circuit Zandvoort (Netherlands)',
        'Autodromo Nazionale Monza (Monza, Italy)', 'MADRING (Madrid, Spain)', 'Baku City Circuit (Azerbaijan)',
        'Marina Bay Street Circuit (Singapore)', 'Circuit of The Americas (Austin, USA)', 'Autódromo Hermanos Rodríguez (Mexico)',
        'Autódromo José Carlos Pace (Brazil)', 'Las Vegas Strip Circuit (Las Vegas, USA)', 'Lusail International Circuit (Qatar)',
        'Yas Marina Circuit (UAE)'
    ]
}

def get_yearly_tracks(years=None):
    """Returns track calendars for the requested seasons."""
    if years is None:
        return YEARLY_TRACKS
    return {year: YEARLY_TRACKS[year] for year in years if year in YEARLY_TRACKS}
