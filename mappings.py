# Mappings for F1 Tire Degradation Model

# Normalization for Constructor/Team Names
# keys: raw names from FastF1/Official sources
# values: canonical name for the model
TEAM_MAPPING = {
    'AlphaTauri': 'Racing Bulls',
    'Scuderia AlphaTauri': 'Racing Bulls',
    'RB': 'Racing Bulls',
    'Visa Cash App RB F1 Team': 'Racing Bulls',
    'Toro Rosso': 'Racing Bulls', # Historical context if needed
    
    'Alfa Romeo': 'Sauber',
    'Alfa Romeo Racing': 'Sauber',
    'Kick Sauber': 'Sauber',
    'Stake F1 Team Kick Sauber': 'Sauber',
    
    'Alpine F1 Team': 'Alpine',
    'BWT Alpine F1 Team': 'Alpine',
    
    'Aston Martin': 'Aston Martin',
    'Aston Martin Aramco Cognizant Formula One Team': 'Aston Martin',
    'Aston Martin Aramco F1 Team': 'Aston Martin',
    
    'Ferrari': 'Ferrari',
    'Scuderia Ferrari': 'Ferrari',
    'Scuderia Ferrari HP': 'Ferrari',
    
    'Haas F1 Team': 'Haas',
    'MoneyGram Haas F1 Team': 'Haas',
    'Haas': 'Haas',
    
    'McLaren': 'McLaren',
    'McLaren F1 Team': 'McLaren',
    
    'Mercedes': 'Mercedes',
    'Mercedes-AMG PETRONAS F1 Team': 'Mercedes',
    
    'Red Bull Racing': 'Red Bull',
    'Oracle Red Bull Racing': 'Red Bull',
    
    'Williams': 'Williams',
    'Williams Racing': 'Williams'
}

# Track Categorization based on speed/characteristics
# Low Speed: Monaco, Singapore, Hungary, Mexico, Zandvoort, Miami
# Medium Speed: Bahrain, Barcelona, Imola, Montreal, COTA, Baku, Austria, Australia, Suzuka, Qatar, Las Vegas
# High Speed: Monza, Silverstone, Spa, Jeddah, Mugello
TRACK_CONFIG = {
    'Bahrain International Circuit': {'type': 'Medium', 'length_km': 5.412},
    'Jeddah Corniche Circuit': {'type': 'High', 'length_km': 6.174},
    'Albert Park Grand Prix Circuit': {'type': 'Medium', 'length_km': 5.278},
    'Baku City Circuit': {'type': 'Medium', 'length_km': 6.003},
    'Miami International Autodrome': {'type': 'Low', 'length_km': 5.412},
    'Circuit de Monaco': {'type': 'Low', 'length_km': 3.337},
    'Circuit de Barcelona-Catalunya': {'type': 'Medium', 'length_km': 4.657},
    'Circuit Gilles Villeneuve': {'type': 'Medium', 'length_km': 4.361},
    'Red Bull Ring': {'type': 'Medium', 'length_km': 4.318},
    'Silverstone Circuit': {'type': 'High', 'length_km': 5.891},
    'Hungaroring': {'type': 'Low', 'length_km': 4.381},
    'Circuit de Spa-Francorchamps': {'type': 'High', 'length_km': 7.004},
    'Circuit Zandvoort': {'type': 'Low', 'length_km': 4.259},
    'Autodromo Nazionale Monza': {'type': 'High', 'length_km': 5.793},
    'Marina Bay Street Circuit': {'type': 'Low', 'length_km': 4.940},
    'Suzuka Circuit': {'type': 'Medium', 'length_km': 5.807}, # User listed as Medium
    'Lusail International Circuit': {'type': 'Medium', 'length_km': 5.419},
    'Circuit of The Americas': {'type': 'Medium', 'length_km': 5.513},
    'Autódromo Hermanos Rodríguez': {'type': 'Low', 'length_km': 4.304},
    'Autódromo José Carlos Pace': {'type': 'Medium', 'length_km': 4.309}, # Interlagos - defaulting to Medium if not specified, but typically medium-low
    'Las Vegas Strip Circuit': {'type': 'Medium', 'length_km': 6.201},
    'Yas Marina Circuit': {'type': 'Medium', 'length_km': 5.281},
    'Autodromo Enzo e Dino Ferrari': {'type': 'Medium', 'length_km': 4.909}, # Imola
    'Shanghai International Circuit': {'type': 'Medium', 'length_km': 5.451},
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
