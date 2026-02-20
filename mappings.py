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
