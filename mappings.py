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


# Track Categorization based on speed/characteristics
# Low Speed: Monaco, Singapore, Hungary, Mexico, Zandvoort, Miami
# Medium Speed: Bahrain, Barcelona, Imola, Montreal, COTA, Baku, Austria, Australia, Suzuka, Qatar, Las Vegas, Brazil, Abu Dhabi
# High Speed: Monza, Silverstone, Spa, Jeddah, Mugello
TRACK_CONFIG = {
    'Bahrain International Circuit': {'type': 'Medium', 'length_km': 5.412},
    'Jeddah Corniche Circuit': {'type': 'High', 'length_km': 6.174},
    'Albert Park Grand Prix Circuit': {'type': 'Medium', 'length_km': 5.278}, # Australia
    'Baku City Circuit': {'type': 'Medium', 'length_km': 6.003},
    'Miami International Autodrome': {'type': 'Low', 'length_km': 5.412},
    'Circuit de Monaco': {'type': 'Low', 'length_km': 3.337},
    'Circuit de Barcelona-Catalunya': {'type': 'Medium', 'length_km': 4.657},
    'Circuit Gilles Villeneuve': {'type': 'Medium', 'length_km': 4.361}, # Montreal
    'Red Bull Ring': {'type': 'Medium', 'length_km': 4.318}, # Austria
    'Silverstone Circuit': {'type': 'High', 'length_km': 5.891},
    'Hungaroring': {'type': 'Low', 'length_km': 4.381}, # Hungary
    'Circuit de Spa-Francorchamps': {'type': 'High', 'length_km': 7.004},
    'Circuit Zandvoort': {'type': 'Low', 'length_km': 4.259},
    'Autodromo Nazionale Monza': {'type': 'High', 'length_km': 5.793},
    'Marina Bay Street Circuit': {'type': 'Low', 'length_km': 4.940}, # Singapore
    'Suzuka Circuit': {'type': 'Medium', 'length_km': 5.807},
    'Lusail International Circuit': {'type': 'Medium', 'length_km': 5.419}, # Qatar
    'Circuit of The Americas': {'type': 'Medium', 'length_km': 5.513}, # COTA
    'Autódromo Hermanos Rodríguez': {'type': 'Low', 'length_km': 4.304}, # Mexico City
    'Autódromo José Carlos Pace': {'type': 'Medium', 'length_km': 4.309}, # Brazil
    'Las Vegas Strip Circuit': {'type': 'Medium', 'length_km': 6.201},
    'Yas Marina Circuit': {'type': 'Medium', 'length_km': 5.281},
    'Autodromo Enzo e Dino Ferrari': {'type': 'Medium', 'length_km': 4.909}, # Imola
    'Shanghai International Circuit': {'type': 'Medium', 'length_km': 5.451},
    'Autodromo Internazionale del Mugello': {'type': 'High', 'length_km': 5.245}
}

def get_track_info(circuit_name):
    """
    Returns track classification and length for a given circuit name.
    """
    # Simple direct lookup or partial match
    if circuit_name in TRACK_CONFIG:
        return TRACK_CONFIG[circuit_name]
    
    # Fallback / Partial match logic could go here
    # For now return defaults
    return {'type': 'Medium', 'length_km': 5.0}

def normalize_team_name(team_name):
    """
    Normalizes team name to a canonical ID.
    """
    return TEAM_MAPPING.get(team_name, team_name)

TRACK_PIT_LOSS = {
    'Bahrain International Circuit': 21.0,
    'Jeddah Corniche Circuit': 20.0,
    'Albert Park Grand Prix Circuit': 21.0,
    'Baku City Circuit': 21.0,
    'Miami International Autodrome': 20.0,
    'Circuit de Monaco': 19.0,
    'Circuit de Barcelona-Catalunya': 22.0,
    'Circuit Gilles Villeneuve': 20.0,
    'Red Bull Ring': 20.0,
    'Silverstone Circuit': 24.0,
    'Hungaroring': 22.0,
    'Circuit de Spa-Francorchamps': 23.0,
    'Circuit Zandvoort': 20.0,
    'Autodromo Nazionale Monza': 24.0,
    'Marina Bay Street Circuit': 26.0,
    'Suzuka Circuit': 22.0,
    'Lusail International Circuit': 21.0,
    'Circuit of The Americas': 22.0,
    'Autódromo Hermanos Rodríguez': 22.0,
    'Autódromo José Carlos Pace': 21.0,
    'Las Vegas Strip Circuit': 20.0,
    'Yas Marina Circuit': 22.0,
    'Autodromo Enzo e Dino Ferrari': 25.0,
    'Shanghai International Circuit': 23.0,
    'Autodromo Internazionale del Mugello': 25.0
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
