import requests
import datetime

# Hardcoded coordinates for the tracks defined in mappings.py
TRACK_COORDINATES = {
    'Bahrain International Circuit (Bahrain)': {'lat': 26.0325, 'lon': 50.5106},
    'Jeddah Corniche Circuit (Saudi Arabia)': {'lat': 21.6319, 'lon': 39.1044},
    'Albert Park Grand Prix Circuit (Australia)': {'lat': -37.8497, 'lon': 144.968},
    'Autodromo Internazionale Enzo e Dino Ferrari (Emilia-Romagna, Italy)': {'lat': 44.3439, 'lon': 11.7167},
    'Miami International Autodrome (Miami, USA)': {'lat': 25.9581, 'lon': -80.2389},
    'Circuit de Barcelona-Catalunya (Spain)': {'lat': 41.57, 'lon': 2.2611},
    'Circuit de Barcelona-Catalunya (Barcelona, Spain)': {'lat': 41.57, 'lon': 2.2611},
    'Circuit de Monaco (Monaco)': {'lat': 43.7347, 'lon': 7.4206},
    'Baku City Circuit (Azerbaijan)': {'lat': 40.3725, 'lon': 49.8533},
    'Circuit Gilles Villeneuve (Canada)': {'lat': 45.5000, 'lon': -73.5228},
    'Silverstone Circuit (Great Britain)': {'lat': 52.0786, 'lon': -1.0169},
    'Red Bull Ring (Austria)': {'lat': 47.2197, 'lon': 14.7647},
    'Circuit Paul Ricard (France)': {'lat': 43.2506, 'lon': 5.7917},
    'Hungaroring (Hungary)': {'lat': 47.5822, 'lon': 19.2511},
    'Circuit de Spa-Francorchamps (Belgium)': {'lat': 50.4372, 'lon': 5.9714},
    'Circuit Zandvoort (Netherlands)': {'lat': 52.3888, 'lon': 4.5409},
    'Autodromo Nazionale Monza (Monza, Italy)': {'lat': 45.6156, 'lon': 9.2811},
    'Marina Bay Street Circuit (Singapore)': {'lat': 1.2914, 'lon': 103.864},
    'Suzuka Circuit (Japan)': {'lat': 34.8431, 'lon': 136.533},
    'Lusail International Circuit (Qatar)': {'lat': 25.4900, 'lon': 51.4542},
    'Circuit of The Americas (Austin, USA)': {'lat': 30.1328, 'lon': -97.6411},
    'Autódromo Hermanos Rodríguez (Mexico)': {'lat': 19.4042, 'lon': -99.0907},
    'Autódromo José Carlos Pace (Brazil)': {'lat': -23.7036, 'lon': -46.6997},
    'Las Vegas Strip Circuit (Las Vegas, USA)': {'lat': 36.1147, 'lon': -115.173},
    'Yas Marina Circuit (UAE)': {'lat': 24.4672, 'lon': 54.6031},
    'Shanghai International Circuit (China)': {'lat': 31.3389, 'lon': 121.222},
    'MADRING (Madrid, Spain)': {'lat': 40.4736, 'lon': -3.6186},
    'Autodromo Internazionale del Mugello (Italy)': {'lat': 43.9975, 'lon': 11.3719}
}

# Cache for weather results to avoid redundant API calls during a single session
_WEATHER_CACHE = {}

def get_track_weather(track_name: str, race_date: str, race_time: str = None) -> dict:
    """
    Fetches weather expected on the race_date using Open-Meteo API.
    Uses in-memory cache to ensure repeated simulations for the same race are instant.
    """
    cache_key = (track_name, race_date, race_time)
    if cache_key in _WEATHER_CACHE:
        return _WEATHER_CACHE[cache_key]
        
    if track_name not in TRACK_COORDINATES:
        print(f"Coordinates for {track_name} not found. Using defaults.")
        return {
            "air_temp": 25.0,
            "track_temp": 35.0,
            "humidity": 50.0,
            "rainfall": False,
            "wind_speed": 2.0,
            "synopsis": "Unknown track coordinates; assuming dry, moderate conditions.",
            "hourly_forecasts": []
        }

    coords = TRACK_COORDINATES[track_name]
    
    # Determine which endpoint to use (archive vs forecast)
    input_date = datetime.date.fromisoformat(race_date)
    today = datetime.date.today()
    
    days_diff = (input_date - today).days
    
    if days_diff > 14:
        # Too far in the future for a forecast. Use historical data from exactly 1 year prior as an estimate.
        target_date = input_date
        while target_date > today - datetime.timedelta(days=5):
            try:
                target_date = target_date.replace(year=target_date.year - 1)
            except ValueError:
                # Handle leap years (Feb 29 -> Feb 28)
                target_date = target_date.replace(year=target_date.year - 1, day=28)
                
        race_date = target_date.isoformat()
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords['lat'],
            "longitude": coords['lon'],
            "start_date": race_date,
            "end_date": race_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "auto"
        }
    elif input_date < today - datetime.timedelta(days=5):
        # Historical Data
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords['lat'],
            "longitude": coords['lon'],
            "start_date": race_date,
            "end_date": race_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "auto"
        }
    else:
        # Forecast / Current Data
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords['lat'],
            "longitude": coords['lon'],
            "start_date": race_date,
            "end_date": race_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "auto"
        }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Take the daytime average (assume 14:00/2PM local API time index for the race)
        # We'll just take the max temp for the day as representative of the race.
        hourly = data.get("hourly", {})
        
        if "temperature_2m" not in hourly:
            raise ValueError("No hourly temperature data found")
            
        temps = hourly["temperature_2m"]
        humidities = hourly["relative_humidity_2m"]
        precips = hourly["precipitation"]
        wind_speeds = hourly["wind_speed_10m"]
        
        # Calculate daily aggregates for fallback/safety
        max_temp = max(t for t in temps if t is not None)
        total_precip = sum(p for p in precips if p is not None)
        is_raining = total_precip > 1.0 # 1mm threshold
        
        # Night races have different temperature profiles
        NIGHT_RACES = {
            'Bahrain International Circuit (Bahrain)',
            'Jeddah Corniche Circuit (Saudi Arabia)',
            'Marina Bay Street Circuit (Singapore)',
            'Las Vegas Strip Circuit (Las Vegas, USA)',
            'Lusail International Circuit (Qatar)',
            'Yas Marina Circuit (UAE)'
        }
        
        is_night_race = track_name in NIGHT_RACES
        hourly_forecasts = []
        
        # Extract temperatures based on explicitly provided time or fallbacks
        if race_time:
            try:
                start_hour = int(race_time.split(":")[0])
                # Grab a 3-hour window
                end_hour = min(23, start_hour + 2)
                
                race_temps = [t for t in temps[start_hour:end_hour+1] if t is not None]
                race_humid = [h for h in humidities[start_hour:end_hour+1] if h is not None]
                race_wind  = [w for w in wind_speeds[start_hour:end_hour+1] if w is not None]
                race_precip = sum(p for p in precips[start_hour:end_hour+1] if p is not None)

                base_temp = sum(race_temps) / len(race_temps) if race_temps else max_temp
                avg_humidity = sum(race_humid) / len(race_humid) if race_humid else 50.0
                avg_wind = sum(race_wind) / len(race_wind) if race_wind else 2.0
                is_raining = race_precip > 1.0
                
                is_night_race = start_hour >= 18 # Override default
                
                # Extract hourly data for the SIM engine
                for hr in range(start_hour, end_hour + 1):
                    hr_air = temps[hr] if temps[hr] is not None else base_temp
                    hr_precip = precips[hr] if precips[hr] is not None else 0.0
                    hr_is_raining = hr_precip > 1.0
                    if is_night_race:
                        hr_track = hr_air + (1.0 if hr_is_raining else 3.0)
                    else:
                        hr_track = hr_air + (5.0 if hr_is_raining else 12.0)
                    
                    hourly_forecasts.append({
                        "hour": hr,
                        "air_temp": round(hr_air, 1),
                        "track_temp": round(hr_track, 1),
                        "rainfall_mm": round(hr_precip, 2)
                    })
                    
            except (ValueError, IndexError):
                race_time = None
                
        if not race_time:
            # Fallback logic
            if is_night_race:
                race_temps = [t for t in temps[19:23] if t is not None]
            else:
                race_temps = [t for t in temps[14:18] if t is not None]
            base_temp = sum(race_temps) / len(race_temps) if race_temps else max_temp
            
            avg_humidity = sum(h for h in humidities if h is not None) / len([h for h in humidities if h is not None])
            avg_wind = sum(w for w in wind_speeds if w is not None) / len([w for w in wind_speeds if w is not None])

        # Track temp estimate
        if is_night_race:
            track_temp_est = base_temp + (1.0 if is_raining else 3.0)
        else:
            track_temp_est = base_temp + (5.0 if is_raining else 12.0)
        
        synopsis = "Expect heavy rain." if ((race_time and race_precip > 5.0) or (not race_time and total_precip > 5.0)) else \
                   "Chance of rain." if is_raining else \
                   "Night race, cool track." if is_night_race and base_temp < 25 else \
                   "Night race, warm track." if is_night_race else \
                   "Hot and clear." if base_temp > 30 else \
                   "Cool and clear." if base_temp < 20 else \
                   "Moderate, dry conditions."
                   
        result = {
            "air_temp": round(base_temp, 1),
            "track_temp": round(track_temp_est, 1),
            "humidity": round(avg_humidity, 1),
            "rainfall": is_raining,
            "wind_speed": round(avg_wind, 1),
            "synopsis": synopsis,
            "hourly_forecasts": hourly_forecasts
        }
        _WEATHER_CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"Error fetching weather data for {track_name}: {e}")
        return {
            "air_temp": 25.0,
            "track_temp": 35.0,
            "humidity": 50.0,
            "rainfall": False,
            "wind_speed": 2.0,
            "synopsis": "API fallback; default conditions assumed.",
            "hourly_forecasts": []
        }
