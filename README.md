# 🏎️ F1 Tire Strategy & Degradation Predictor

## [Launch the Live Application](https://f1-tire-deg.netlify.app/)

A high-performance Formula 1 tire-degradation inference and race-strategy optimization platform. The system combines era-specific gradient-boosted regression models, circuit-level feature engineering, weather-aware simulation, and constrained strategy search to produce compound degradation profiles and race-state-aware pit recommendations.

The application provides two primary analytical outputs:

- **Compound Degradation Forecasting**: Predicts tire wear and lap-time evolution for each available compound at a selected circuit, conditioned on driver, constructor, technical era, and environmental inputs.
- **Race Strategy Optimization**: Searches valid compound sequences and stint allocations using the predicted degradation curves, current race state, circuit-specific pit loss, safety-car state, and live or historical weather data.

---

## 🌟 Core Features

### 1. Compound-Specific Degradation Inference

The inference engine generates full-race degradation curves for Soft, Medium, Hard, Intermediate, and Wet compounds. Each curve is conditioned on:

- Driver and team
- Circuit type, length, surface, and stress characteristics
- Tire age and normalized tire life
- Air temperature, estimated track temperature, humidity, rainfall, and wind
- Compound-specific interactions with abrasion, traction, thermal stress, and tire age

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

### 5. Advanced Track Feature Engineering

Circuit behavior is represented through a normalized 10-dimensional relational feature system:

- **Raw Characteristics**: Traction intensity, High-speed load, Abrasiveness (surface texture), Surface roughness (micro-profile), Braking severity, Lateral load, and Track temperature sensitivity.
- **Derived Stress Metrics**: Weighted blends for **Thermal Stress**, **Surface Wear**, and **Total Energy Load**.
- **Cross-Era Normalization**: Features are normalized to `[0, 1]`, enabling transferable interaction learning between circuit stress, compound choice, and tire age.

### 6. Real-Time Strategy Optimization

- **Constrained Strategy Search**: Evaluates compound permutations and bounded stint-length combinations while enforcing compound diversity and physical stint caps.
- **Weather Integration**: Live and historical weather data (air/track temp, humidity, rainfall) via Open-Meteo API to adjust degradation curves in real-time.
- **Safety Car Modeling**: Applies reduced tire-wear accumulation and discounted pit-loss estimates during active safety-car scenarios.

### 7. Interactive Dashboard

A React-based analytical dashboard provides:

- Multi-compound lap-time degradation curves rendered with Recharts.
- Track-stress metrics, weather forecasts, tire-life diagnostics, and degradation-rate summaries.
- Dynamic stint bars for optimal, safe, and risky strategy comparisons.

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
- Ten normalized circuit characteristics
- Interaction features such as tire age × abrasiveness, tire age × traction, tire age × lateral load, and temperature × track sensitivity
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
- **API**: Flask
- **Data**: FastF1 API integration

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Visualization**: Recharts
- **Aesthetic**: Custom "Carbon Black" CSS design system

---

## 📂 Project Structure

```text
├── app.py                  # Flask API Entry Point
├── degradation_engine.py    # Core ML-based degradation simulation
├── tire_life_analysis.py    # Performance cliff & useful life logic
├── sim_engine.py           # Race simulation & strategy generation
├── strategy_optimizer.py    # Multi-path optimization algorithms
├── train_era_models.py      # Era-specific ML model trainer
├── weather_api.py           # Open-Meteo integration & caching
├── frontend/               # React + Vite Dashboard
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

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

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
2. Enter the race distance and current race-state information.
3. Add tire-age, pit-history, and safety-car context when simulating mid-race decisions.
4. Run the simulation to compare degradation curves and strategy recommendations.

---

## 🔧 Automation
The project includes a single GitHub Actions workflow for weekly 2026+ era retraining:

```text
.github/workflows/weekly-model-retrain.yml
```

It runs every Monday at 03:00 EST (`0 8 * * 1` in GitHub's UTC cron). During daylight saving time, the same fixed UTC schedule runs at 04:00 EDT.

Before training, the workflow runs `scripts/check_active_aero_retrain_needed.py`. The preflight reads `models/era_training_metadata.json`, checks FastF1's completed race schedule for the Active Aero era, and only allows retraining when a completed race is missing from the active model metadata. If no new completed race data is available, the workflow skips training and records the reason in the GitHub Actions step summary. If FastF1 schedule lookup fails, the workflow fails visibly instead of silently treating the run as "no update needed."

Manual workflow dispatch includes a `force_retrain` option for deliberately running `scripts/retrain_active_aero_weekly.sh` even when the preflight says the model is current. Completed training attempts continue to update `models/era_training_metadata.json` with loaded and failed FastF1 sessions.

A temporary test workflow also exists at `.github/workflows/weekly-model-retrain-test.yml`. It is scheduled once for Friday, July 3, 2026 at 2:00 PM EDT (`0 18 3 7 *` UTC) and directly forces the same Active Aero retraining script. Remove this file after the test run is verified.

### Keep the backend awake on Render
Render will spin down the service after 15 minutes of inactivity. The Netlify Scheduled Function at `frontend/netlify/functions/render-keepalive.cjs` pings the lightweight backend health endpoint every 10 minutes:

```text
https://f1-tire-deg-predictor.onrender.com/health
```

If the Render service URL changes, set a Netlify environment variable named `RENDER_HEALTH_URL` to the new health endpoint. The GitHub Actions workflow at `.github/workflows/render-keepalive.yml` remains as a backup ping, but GitHub cron scheduling is not frequent enough to be the primary keepalive mechanism for Render's 15-minute idle window.
