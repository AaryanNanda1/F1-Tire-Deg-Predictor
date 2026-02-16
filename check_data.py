import fastf1
import pandas as pd

# Enable cache
fastf1.Cache.enable_cache('cache')

# Load a sample session
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

# Check Laps columns
print("Laps Columns:", session.laps.columns.tolist())

# Check Weather columns
print("Weather Columns:", session.weather_data.columns.tolist())

# Check if we can get circuit info easily (length)
# Usually session.event has some info, or we might need a lookup
print("Event Info:", session.event)
