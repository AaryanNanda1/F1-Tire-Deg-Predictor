import fastf1
import os

# Create cache directory if it doesn't exist
# Create cache directory if it doesn't exist
# This is important to avoid re-downloading large data files from the API every time we run the code.
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Enable caching for FastF1 library. 
# Data downloaded from the API will be stored in the 'cache' folder.
fastf1.Cache.enable_cache(CACHE_DIR)

def load_race_data(year, grand_prix, session_type='R'):
    """
    Loads race data from FastF1 API.
    
    The FastF1 library scrapes data from F1's live timing service and other sources.
    This function retrieves the session object which contains lap times, telemetry, and weather data.
    
    Args:
        year (int): Season year (e.g., 2023).
        grand_prix (str): Name of the Grand Prix (e.g., 'Bahrain').
        session_type (str): Session identifier (default 'R' for Race).
        
    Returns:
        fastf1.core.Session: The loaded session object with laps data. 
                             This object is the primary data source for our analysis.
    """
    # fastf1.get_session creates a session object but doesn't download data yet.
    session = fastf1.get_session(year, grand_prix, session_type)
    
    # session.load() actually downloads the data from the API (or loads from cache).
    # This includes lap times, car telemetry, weather, etc.
    # we explicitly request 'weather' and 'laps' though they are loaded by default.
    session.load()
    
    return session

if __name__ == "__main__":
    # Test the loader
    print("Testing data loader...")
    try:
        session = load_race_data(2023, 'Bahrain')
        print(f"Loaded {session.event['EventName']} - {len(session.laps)} laps")
    except Exception as e:
        print(f"Error loading data: {e}")
