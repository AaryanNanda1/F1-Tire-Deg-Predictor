# 🏎️ F1 Tire Strategy & Degradation Predictor

## [Launch the Live Application](https://f1-tire-deg.netlify.app/)

A high-performance Formula 1 tire-degradation inference and race-strategy optimization platform. The system combines era-specific gradient-boosted regression models, circuit-level feature engineering, weather-aware simulation, and constrained strategy search to produce compound degradation profiles and race-state-aware pit recommendations.

The application provides two primary analytical outputs:

- **Compound Degradation Forecasting**: Predicts tire wear and lap-time evolution for each available compound at a selected circuit, conditioned on driver, constructor, technical era, and environmental inputs.
- **Race Strategy Optimization**: Searches valid compound sequences and stint allocations using the predicted degradation curves, current race state, circuit-specific pit loss, safety-car state, and date-aware Open-Meteo weather context.

---

## 🌟 Core Features

### 1. Compound-Specific Degradation Inference

The inference engine generates full-race degradation curves for Soft, Medium, Hard, Intermediate, and Wet compounds. Each curve is conditioned on:

- Driver and team
- Circuit type, length, surface, and stress characteristics
- Tire age and normalized tire life
- Air temperature, estimated track temperature, humidity, rainfall, and wind
- Compound-specific interactions with sourced abrasion, traction, tyre stress, and tire age

### 2. Race-State-Aware Strategy Optimization

The strategy engine performs a constrained search across compound permutations and stint-length allocations. Candidate strategies are evaluated against:

- Total race distance and current lap
- Current compound and tire age
- Grid or track position
- Previous pit stops and compounds already used
- Safety-car laps and reduced safety-car pit loss
- Circuit-specific pit loss and overtaking penalties
- Dry, intermediate, and wet-weather conditions

The optimizer returns the minimum predicted race-time strategy and, when materially distinct, lower-risk and higher-risk alternatives. Strategy outputs include compound sequences, lap-indexed stint windows, stop counts, degradation cost, and cliff-overshoot exposure.

### 3. Dual-Output Tire Life Engine

The post-processing pipeline derives two independent tire-life metrics from each predicted curve:

- **Performance Cliff Detection**: Uses a sustained-acceleration rule over smoothed slope and curvature signals to identify the onset of nonlinear lap-time deterioration.
- **Strategic Useful Life**: Computes the crossover point where cumulative degradation loss exceeds the circuit-specific cost of replacing the tire.

### 4. Multi-Era Modeling

Separate model artifacts isolate materially different aerodynamic and tire-regulation regimes:

- **Ground Effect Era (2022–2025)**: Trained on historical long-run telemetry from race, Sprint, and FP2 sessions.
- **Active Aero Era (2026–2030)**: Uses a separate model for the new technical regulations.
- **Hybrid Cold Start Methodology**: Stabilizes sparse new-era training by blending available 2026 telemetry with weighted prior-era observations or synthetic prior-model predictions.

### 5. Source-Backed Track Feature Engineering

Circuit behavior is represented through seven reproducible normalized features:

- **Pirelli ratings**: Traction, Tyre Stress, Asphalt Grip, Braking, Asphalt Abrasion, and Lateral are transcribed from each event's official 2025 Track Characteristics graphic. A published rating `r` on the 1–5 scale is normalized as `(r - 1) / 4`.
- **Mercedes corner-speed energy**: Minimum speeds printed on the official track maps are transformed as the mean of `(min(speed, 300) / 300)²`, giving high-speed turns more influence without presenting the result as a physical force measurement.
- **Auditable source data**: Every raw rating, turn speed, article URL, graphic URL, and Mercedes asset page is stored in [`data/track_characteristics_2025.csv`](data/track_characteristics_2025.csv). No source images are copied into the repository.

The previous uncited weighted composites and unsupported surface/temperature labels have been removed. See the complete [track-characteristic methodology](docs/track_characteristics_methodology.md), including missing-data rules and limitations.

### 6. Date-Aware Weather Context

The dashboard accepts a race date and local start time for every simulation. When a selected circuit and season resolve to a FastF1 event, the fields are prefilled with the scheduled race start and remain user-editable.

Weather retrieval is scoped to the selected circuit coordinates and exact race date:

- **Forecast horizon**: Dates from today through 16 days ahead use Open-Meteo's hourly forecast endpoint.
- **Historical races**: Completed dates more than five days old use Open-Meteo's historical archive.
- **Long-range simulations**: Dates beyond the forecast horizon use weather from approximately the same date in a prior year as a clearly defined seasonal estimate.
- **Caching**: Forecast results expire after one hour; historical and seasonal results are cached for 30 days.

The weather context passed to the degradation model includes air temperature, humidity, precipitation, and wind. Track temperature is a deterministic estimate derived from air temperature: daytime dry conditions add 12°C, daytime wet conditions add 5°C, night dry conditions add 3°C, and night wet conditions add 1°C. These offsets are modeling assumptions, not measurements returned by Open-Meteo.

### 7. Real-Time Strategy Optimization

- **Constrained Strategy Search**: Evaluates compound permutations and bounded stint-length combinations while enforcing the mandatory pit-stop rule, wet-weather compound exceptions, and physical stint caps.
- **Weather Integration**: Selected-date forecast, historical, or seasonal-estimate weather context via the [Open-Meteo API](https://open-meteo.com/en/docs), used to adjust degradation curves.
- **Safety Car Modeling**: Applies reduced tire-wear accumulation and discounted pit-loss estimates during active safety-car scenarios.

### 8. Interactive Dashboard

A React-based analytical dashboard provides:

- Multi-compound lap-time degradation curves rendered with Recharts.
- FastF1 calendar-backed race date/time defaults, editable weather timing inputs, track-stress metrics, weather conditions, tire-life diagnostics, and degradation-rate summaries.
- Dynamic stint bars for optimal, safe, and risky strategy comparisons.
- Race-state input guardrails that auto-correct invalid lap, tire-age, safety-car, and track-position combinations without changing the input style.

---

## 🧠 Machine Learning Methodology

### Model Architecture

Each technical era uses an independent Scikit-Learn `HistGradientBoostingRegressor` configured with absolute-error loss. Histogram-based gradient boosting provides efficient nonlinear regression over mixed operational inputs while remaining robust to the asymmetric noise and outliers common in motorsport telemetry.

The supervised target is **lap-time delta relative to the constructor baseline**, rather than absolute lap time. This target transformation reduces circuit-identity leakage and directs model capacity toward degradation behavior. During inference, the predicted delta is reconstructed against the circuit baseline to obtain the absolute compound curves displayed by the dashboard.

### Training Data and Weighting

Training data is collected through FastF1 and restricted to representative long-run session types:

- Race laps receive a sample weight of `1.00`.
- Sprint laps receive a sample weight of `0.75`.
- FP2 laps receive a sample weight of `0.50`.
- Compound-specific weighting and telemetry outlier filtering reduce the influence of nonrepresentative laps, traffic artifacts, and anomalous stint behavior.

The `--as-of-date` boundary enforces chronological data availability, allowing point-in-time model reconstruction and backtesting without future-event leakage.

### Feature Engineering

The training pipeline combines:

- Tire age, stint length, normalized tire life, and squared tire age
- Fuel-load and race-distance context
- Driver, team, compound, track type, and event categorical features
- Weather conditions, including air and track temperature, humidity, rainfall, and wind
- Seven source-backed circuit characteristics
- Interaction features such as tire age × abrasiveness, tire age × traction, tire age × lateral load, temperature × tyre stress, and normalized tire life × tyre stress
- Compound-specific age and circuit interactions

Categorical values are one-hot encoded with fixed category domains. Each model persists its training feature schema, and inference matrices are explicitly reindexed against that schema to prevent training-serving skew.

### Multi-Era and Cold-Start Strategy

Era segmentation prevents the estimator from treating major aerodynamic and tire-regulation changes as a stationary process. When current-era telemetry is sparse, the training pipeline blends contemporary observations with weighted prior-era samples. An optional synthetic-prior path can also project current-era feature vectors through the preceding model and inject the resulting targets at reduced sample weight.

### Validation and Post-Processing

Walk-forward validation orders events chronologically, trains only on preceding rounds, and evaluates the next event. This approximates in-season generalization more accurately than a random row-level split. During inference:

1. The model predicts a lap-time delta for each tire age and compound.
2. Predictions are converted into absolute lap-time curves.
3. Nonphysical local reversals are removed through monotonic degradation enforcement.
4. Smoothed slope and curvature analysis estimates the physical performance cliff.
5. Integrated degradation cost is compared with circuit pit loss to estimate strategic useful life.

The machine-learning layer is deliberately isolated from strategy policy. It produces degradation primitives; the deterministic strategy engine consumes those primitives and evaluates race-level decisions under explicit sporting and operational constraints.

---

## 🛠️ Tech Stack

### Backend
- **Core**: Python 3.10+
- **Analysis**: NumPy, Pandas, SciPy, Scikit-Learn
- **API**: Flask + Flask-CORS
- **Data**: FastF1 event schedules and telemetry; Open-Meteo weather API; Pirelli Track Characteristics ratings; Mercedes-AMG F1 track maps
- **Deployment**: Render with Gunicorn (`gunicorn --timeout 120 app:app`)

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Visualization**: Recharts
- **Aesthetic**: Custom "Carbon Black" CSS design system
- **Hosting**: Netlify

---

## 📂 Project Structure

```text
├── .github/workflows/      # GitHub Actions automation
├── app.py                  # Flask API Entry Point
├── degradation_engine.py    # Core ML-based degradation simulation
├── tire_life_analysis.py    # Performance cliff & useful life logic
├── sim_engine.py           # Race simulation & strategy generation
├── strategy_optimizer.py    # Multi-path optimization algorithms
├── train_era_models.py      # Era-specific ML model trainer
├── weather_api.py           # Open-Meteo integration & caching
├── render.yaml             # Render deployment config
├── frontend/               # React + Vite Dashboard and Netlify config
└── scripts/                # Retraining & automation utilities
```

---

## 🚀 Setup & Installation

### 1. Backend Setup
```bash
# Clone the repository
git clone <repository-url>
cd F1-Tire-Deg-Predictor

# Install Python dependencies
pip install -r requirements.txt

# Start the API server (Port 5001)
python app.py
```

Production runs the backend with Render's Gunicorn command:

```bash
gunicorn --timeout 120 app:app
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Local Vite development proxies `/api/*` requests to `http://127.0.0.1:5001`. Production defaults directly to the Render backend at `https://f1-tire-deg-predictor.onrender.com` to avoid Netlify proxy timeouts on long simulations. Netlify still keeps an `/api/*` redirect fallback. Override the API origin with `VITE_API_BASE_URL` when needed.

### 3. Model Training
To train era-specific models (Ground Effect & Active Aero):
```bash
python train_era_models.py --mode both --as-of-date 2026-02-20 --output-dir models
```

---

## 📊 Usage

### CLI Strategy Planner
Run a pre-race strategy optimization via command line:
```bash
python run_strategy.py \
  --year 2024 \
  --gp Bahrain \
  --driver VER \
  --team "Red Bull Racing" \
  --race-laps 57 \
  --condition auto
```

### Dashboard

Use the [deployed dashboard](https://f1-tire-deg.netlify.app/) or run the frontend locally at `http://localhost:3000`.

1. Select the race year, track, team, and driver.
2. Confirm or edit the FastF1 calendar-defaulted race date and local start time. If FastF1 cannot resolve the event, choose the timing manually.
3. Enter the race distance and current race-state information.
4. Add tire-age, pit-history, and safety-car context when simulating mid-race decisions.
5. Run the simulation to compare degradation curves and strategy recommendations.

For dates inside the 16-day forecast horizon, the simulation uses the selected date's Open-Meteo forecast. Historical dates use archived conditions; simulations beyond the forecast horizon use a prior-year seasonal estimate rather than a weather forecast.

Race-state inputs are normalized automatically:

- Current lap cannot exceed total race laps.
- Laps on the current tire cannot exceed the current lap.
- Safety-car laps on the current tire cannot exceed laps on the current tire.
- At lap 0, tire age and safety-car tire laps are forced to 0, and track position is forced to grid position.
- If the driver has already pitted, laps on the current tire must be less than the current lap.
- If the driver has already pitted, distinct compounds used is collected and clamped between 1 and 5.

### Tire Strategy Rules

Every generated strategy must include at least one physical pit stop unless the driver has already pitted earlier in the race. The dry-race two-compound requirement is waived when an Intermediate or Wet tire is used at any point in the race. As a result, same-compound race plans are only valid when they are entirely Intermediate or entirely Wet.

Weather-specific strategy behavior:

- Dry conditions search dry compounds and penalize Intermediate or Wet tires.
- Light-wet conditions search dry compounds and Intermediates, with Full Wet tires allowed only in controlled Intermediate/Wet transition patterns.
- Heavy-wet conditions search Intermediate and Wet tires.
- Dry tires receive a rain penalty in wet conditions; Soft loses the least lap time, Medium sits between them, and Hard loses the most.

### API & Deployment

The backend exposes health metadata at both `/health` and `/api/health`:

```json
{
  "status": "ok",
  "service": "f1-tire-deg-predictor",
  "version": {
    "commit": "full git sha or unknown",
    "commit_short": "short sha",
    "source": "render, local-git, or unknown"
  }
}
```

Use this endpoint to confirm which commit Render is currently running. Render provides `RENDER_GIT_COMMIT` in production; local runs fall back to `git rev-parse HEAD` when available.

The backend allows CORS from the Netlify production app and local Vite development origins by default. Set `CORS_ORIGINS` on Render as a comma-separated list if the frontend origin changes.

---

## 🔧 Automation
The project includes a single GitHub Actions workflow for weekly 2026+ era retraining:

```text
.github/workflows/weekly-model-retrain.yml
```

It runs every Monday at 03:00 EST (`0 8 * * 1` in GitHub's UTC cron). During daylight saving time, the same fixed UTC schedule runs at 04:00 EDT.

Before training, the workflow runs `scripts/check_active_aero_retrain_needed.py`. The preflight reads `models/era_training_metadata.json`, checks FastF1's completed race schedule for the Active Aero era, and only allows retraining when a completed race is missing from the active model metadata. If no new completed race data is available, the workflow skips training and records the reason in the GitHub Actions step summary. If FastF1 schedule lookup fails, the workflow fails visibly instead of silently treating the run as "no update needed."

Manual workflow dispatch includes a `force_retrain` option for deliberately running `scripts/retrain_active_aero_weekly.sh` even when the preflight says the model is current. Completed training attempts continue to update `models/era_training_metadata.json` with loaded and failed FastF1 sessions.

### Keep the backend awake on Render
Render will spin down the service after 15 minutes of inactivity. The Netlify Scheduled Function at `frontend/netlify/functions/render-keepalive.cjs` pings the lightweight backend health endpoint every 10 minutes:

```text
https://f1-tire-deg-predictor.onrender.com/health
```

If the Render service URL changes, set a Netlify environment variable named `RENDER_HEALTH_URL` to the new health endpoint. This scheduled function uses Netlify function compute, not production deploy credits. The GitHub Actions workflow at `.github/workflows/render-keepalive.yml` remains as a backup ping, but GitHub cron scheduling is not frequent enough to be the primary keepalive mechanism for Render's 15-minute idle window.
