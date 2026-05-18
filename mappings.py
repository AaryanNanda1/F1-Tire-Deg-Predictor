# Mappings for F1 Tire Degradation Model

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
# TRACK CHARACTERISTICS — normalized 0.0–1.0 feature priors
# These are relational/contextual features that help the ML model
# understand how degradation differs between circuits. The model
# still relies primarily on observed data (lap times, tire age, etc.)
# ============================================================
TRACK_CHARACTERISTICS = {
    'Bahrain International Circuit (Bahrain)': {
        'traction': 0.78, 'high_speed_load': 0.58, 'abrasiveness': 0.92,
        'surface_roughness': 0.74, 'braking_severity': 0.82,
        'lateral_load': 0.71, 'track_temp_sensitivity': 0.88
    },
    'Jeddah Corniche Circuit (Saudi Arabia)': {
        'traction': 0.42, 'high_speed_load': 0.96, 'abrasiveness': 0.48,
        'surface_roughness': 0.37, 'braking_severity': 0.51,
        'lateral_load': 0.93, 'track_temp_sensitivity': 0.54
    },
    'Albert Park Grand Prix Circuit (Australia)': {
        'traction': 0.63, 'high_speed_load': 0.67, 'abrasiveness': 0.46,
        'surface_roughness': 0.43, 'braking_severity': 0.66,
        'lateral_load': 0.64, 'track_temp_sensitivity': 0.58
    },
    'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)': {
        'traction': 0.69, 'high_speed_load': 0.73, 'abrasiveness': 0.63,
        'surface_roughness': 0.58, 'braking_severity': 0.71,
        'lateral_load': 0.76, 'track_temp_sensitivity': 0.69
    },
    'Miami International Autodrome (Miami, USA)': {
        'traction': 0.74, 'high_speed_load': 0.54, 'abrasiveness': 0.39,
        'surface_roughness': 0.61, 'braking_severity': 0.72,
        'lateral_load': 0.49, 'track_temp_sensitivity': 0.81
    },
    'Circuit de Barcelona-Catalunya (Spain)': {
        'traction': 0.58, 'high_speed_load': 0.91, 'abrasiveness': 0.88,
        'surface_roughness': 0.52, 'braking_severity': 0.59,
        'lateral_load': 0.95, 'track_temp_sensitivity': 0.84
    },
    'Circuit de Barcelona-Catalunya (Barcelona, Spain)': {
        'traction': 0.58, 'high_speed_load': 0.91, 'abrasiveness': 0.88,
        'surface_roughness': 0.52, 'braking_severity': 0.59,
        'lateral_load': 0.95, 'track_temp_sensitivity': 0.84
    },
    'Circuit de Monaco (Monaco)': {
        'traction': 0.97, 'high_speed_load': 0.18, 'abrasiveness': 0.21,
        'surface_roughness': 0.79, 'braking_severity': 0.58,
        'lateral_load': 0.29, 'track_temp_sensitivity': 0.52
    },
    'Baku City Circuit (Azerbaijan)': {
        'traction': 0.84, 'high_speed_load': 0.56, 'abrasiveness': 0.32,
        'surface_roughness': 0.67, 'braking_severity': 0.89,
        'lateral_load': 0.46, 'track_temp_sensitivity': 0.57
    },
    'Circuit Gilles Villeneuve (Canada)': {
        'traction': 0.79, 'high_speed_load': 0.62, 'abrasiveness': 0.41,
        'surface_roughness': 0.56, 'braking_severity': 0.91,
        'lateral_load': 0.53, 'track_temp_sensitivity': 0.49
    },
    'Silverstone Circuit (Great Britain)': {
        'traction': 0.38, 'high_speed_load': 0.99, 'abrasiveness': 0.76,
        'surface_roughness': 0.47, 'braking_severity': 0.44,
        'lateral_load': 1.00, 'track_temp_sensitivity': 0.83
    },
    'Red Bull Ring (Austria)': {
        'traction': 0.76, 'high_speed_load': 0.63, 'abrasiveness': 0.59,
        'surface_roughness': 0.42, 'braking_severity': 0.94,
        'lateral_load': 0.57, 'track_temp_sensitivity': 0.74
    },
    'Circuit Paul Ricard (France)': {
        'traction': 0.52, 'high_speed_load': 0.81, 'abrasiveness': 0.67,
        'surface_roughness': 0.34, 'braking_severity': 0.63,
        'lateral_load': 0.84, 'track_temp_sensitivity': 0.77
    },
    'Hungaroring (Hungary)': {
        'traction': 0.91, 'high_speed_load': 0.33, 'abrasiveness': 0.61,
        'surface_roughness': 0.53, 'braking_severity': 0.52,
        'lateral_load': 0.73, 'track_temp_sensitivity': 0.89
    },
    'Circuit de Spa-Francorchamps (Belgium)': {
        'traction': 0.49, 'high_speed_load': 0.97, 'abrasiveness': 0.71,
        'surface_roughness': 0.64, 'braking_severity': 0.55,
        'lateral_load': 0.94, 'track_temp_sensitivity': 0.62
    },
    'Circuit Zandvoort (Netherlands)': {
        'traction': 0.74, 'high_speed_load': 0.76, 'abrasiveness': 0.68,
        'surface_roughness': 0.48, 'braking_severity': 0.41,
        'lateral_load': 0.88, 'track_temp_sensitivity': 0.79
    },
    'Autodromo Nazionale Monza (Monza, Italy)': {
        'traction': 0.43, 'high_speed_load': 0.92, 'abrasiveness': 0.39,
        'surface_roughness': 0.31, 'braking_severity': 0.96,
        'lateral_load': 0.46, 'track_temp_sensitivity': 0.41
    },
    'Marina Bay Street Circuit (Singapore)': {
        'traction': 0.95, 'high_speed_load': 0.22, 'abrasiveness': 0.44,
        'surface_roughness': 0.83, 'braking_severity': 0.73,
        'lateral_load': 0.58, 'track_temp_sensitivity': 0.97
    },
    'Suzuka Circuit (Japan)': {
        'traction': 0.51, 'high_speed_load': 0.94, 'abrasiveness': 0.73,
        'surface_roughness': 0.49, 'braking_severity': 0.46,
        'lateral_load': 0.97, 'track_temp_sensitivity': 0.81
    },
    'Lusail International Circuit (Qatar)': {
        'traction': 0.46, 'high_speed_load': 0.93, 'abrasiveness': 0.66,
        'surface_roughness': 0.29, 'braking_severity': 0.37,
        'lateral_load': 0.96, 'track_temp_sensitivity': 0.98
    },
    'Circuit of The Americas (Austin, USA)': {
        'traction': 0.71, 'high_speed_load': 0.78, 'abrasiveness': 0.72,
        'surface_roughness': 0.88, 'braking_severity': 0.81,
        'lateral_load': 0.79, 'track_temp_sensitivity': 0.76
    },
    'Autódromo Hermanos Rodríguez (Mexico)': {
        'traction': 0.82, 'high_speed_load': 0.44, 'abrasiveness': 0.36,
        'surface_roughness': 0.47, 'braking_severity': 0.74,
        'lateral_load': 0.42, 'track_temp_sensitivity': 0.72
    },
    'Autódromo José Carlos Pace (Brazil)': {
        'traction': 0.67, 'high_speed_load': 0.81, 'abrasiveness': 0.57,
        'surface_roughness': 0.73, 'braking_severity': 0.66,
        'lateral_load': 0.83, 'track_temp_sensitivity': 0.71
    },
    'Las Vegas Strip Circuit (Las Vegas, USA)': {
        'traction': 0.57, 'high_speed_load': 0.87, 'abrasiveness': 0.18,
        'surface_roughness': 0.26, 'braking_severity': 0.78,
        'lateral_load': 0.39, 'track_temp_sensitivity': 0.93
    },
    'Yas Marina Circuit (UAE)': {
        'traction': 0.72, 'high_speed_load': 0.52, 'abrasiveness': 0.43,
        'surface_roughness': 0.36, 'braking_severity': 0.69,
        'lateral_load': 0.51, 'track_temp_sensitivity': 0.73
    },
    'Shanghai International Circuit (China)': {
        'traction': 0.66, 'high_speed_load': 0.84, 'abrasiveness': 0.69,
        'surface_roughness': 0.51, 'braking_severity': 0.71,
        'lateral_load': 0.86, 'track_temp_sensitivity': 0.74
    },
    'Autodromo Internazionale del Mugello (Italy)': {
        'traction': 0.36, 'high_speed_load': 1.00, 'abrasiveness': 0.74,
        'surface_roughness': 0.57, 'braking_severity': 0.41,
        'lateral_load': 0.99, 'track_temp_sensitivity': 0.78
    },
    'MADRING (Madrid, Spain)': {
        'traction': 0.83, 'high_speed_load': 0.41, 'abrasiveness': 0.48,
        'surface_roughness': 0.62, 'braking_severity': 0.77,
        'lateral_load': 0.44, 'track_temp_sensitivity': 0.79
    },
}

# Default characteristics for unknown tracks
_DEFAULT_CHARACTERISTICS = {
    'traction': 0.50, 'high_speed_load': 0.50, 'abrasiveness': 0.50,
    'surface_roughness': 0.50, 'braking_severity': 0.50,
    'lateral_load': 0.50, 'track_temp_sensitivity': 0.50
}

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


# ============================================================
# DERIVED FEATURE FUNCTIONS
# ============================================================

def compute_thermal_stress(features):
    """Weighted blend of temperature-related tire stress factors."""
    return (
        0.35 * features['track_temp_sensitivity'] +
        0.30 * features['lateral_load'] +
        0.20 * features['abrasiveness'] +
        0.15 * features['traction']
    )

def compute_surface_wear(features):
    """Weighted blend of surface-induced tire wear factors."""
    return (
        0.50 * features['abrasiveness'] +
        0.30 * features['surface_roughness'] +
        0.20 * features['braking_severity']
    )

def compute_energy_load(features):
    """Weighted blend of total tire energy load from cornering and speed."""
    return (
        0.45 * features['lateral_load'] +
        0.35 * features['high_speed_load'] +
        0.20 * features['traction']
    )


def get_track_features(track_name):
    """
    Returns the full 10-dimensional feature vector for a track:
    7 raw characteristics + 3 derived composites.
    Falls back to neutral defaults for unknown tracks.
    """
    key = resolve_circuit_key(track_name)
    raw = TRACK_CHARACTERISTICS.get(key, _DEFAULT_CHARACTERISTICS).copy()
    raw['thermal_stress'] = compute_thermal_stress(raw)
    raw['surface_wear'] = compute_surface_wear(raw)
    raw['energy_load'] = compute_energy_load(raw)
    return raw


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

def get_roster_map():
    """Generates a full mapping of year -> team -> [drivers] for all supported years."""
    full_map = {}
    
    for year in range(2022, 2031):
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

def get_yearly_tracks():
    return YEARLY_TRACKS
