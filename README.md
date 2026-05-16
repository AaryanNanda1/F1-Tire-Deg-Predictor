# 🏎️ F1 Tire Strategy & Degradation Predictor

![F1 Tire Strategy Dashboard](/Users/aaryannanda/.gemini/antigravity/brain/82f7d8d7-b465-43a6-a1fd-0db6b6350798/f1_dashboard_screenshot_1777353609861.png)

A high-performance, data-driven analysis engine and dashboard for predicting Formula 1 tire degradation and optimizing race strategies. This project combines machine learning with physical simulation to provide insights across different F1 technical eras.

---

## 🌟 Core Features

### 1. Dual-Output Tire Life Engine
Unlike traditional heuristic models, our engine produces two independent, high-fidelity metrics:
- **Performance Cliff Detection**: Identifies the exact lap where a tire's physics begins to fail. Uses a *Sustained Acceleration Rule* (slope and curvature analysis) to detect when lap times begin to worsen uncontrollably.
- **Strategic Useful Life**: Calculates the "crossover point" where the cumulative time lost to degradation exceeds the time lost in a pit stop.

### 2. Multi-Era Modeling & Hybrid Training
Supports disparate aerodynamic and tire regulations across F1 history and future:
- **Ground Effect Era (2022–2025)**: Tuned for high-downforce technical regulations using historical telemetry from over 100 sessions.
- **Active Aero Era (2026–2030)**: Pre-emptive modeling for the upcoming technical overhaul.
- **Hybrid Cold Start Methodology**: To solve data sparsity for new regulations (like 2026), the engine uses a "Hybrid Cold Start" approach, blending real-time telemetry from inaugural races with physics-based priors from previous eras to ensure stable predictions.

### 3. Advanced Track Feature Engineering
The engine now understands *why* tires degrade differently at each circuit through a 10-dimensional relational feature system:
- **Raw Characteristics**: Traction intensity, High-speed load, Abrasiveness (surface texture), Surface roughness (micro-profile), Braking severity, Lateral load, and Track temperature sensitivity.
- **Derived Stress Metrics**: Weighted blends for **Thermal Stress**, **Surface Wear**, and **Total Energy Load**.
- **Cross-Era Normalization**: Features are normalized (0.0–1.0) to allow the model to learn relational patterns (e.g., "how much does abrasiveness accelerate wear on a Soft compound?") that transfer across different car regulations.

### 4. Advanced Training Methodology
Built for robust "in-season" performance and backtesting:
- **FastF1-to-UI Bridge**: A standardized mapping layer resolves disparate naming conventions (FastF1 EventNames vs. UI Track Keys) to ensure seamless data flow from training to inference.
- **Chronological Filtering (`--as-of-date`)**: Simulates a specific point in time by only training on races completed before the target date.
- **Walk-Forward Validation**: An iterative evaluation loop across up to 80 events per era to ensure high generalized accuracy.
- **Session-Weighted Learning**: Dynamically weights data based on session type (Race=1.0, Sprint=0.75, FP2=0.5) to prioritize the most representative long-run data.

### 5. Real-Time Strategy Optimization
- **Monte Carlo Simulations**: Generates thousands of potential strategy permutations to find the optimal pit windows.
- **Weather Integration**: Live and historical weather data (air/track temp, humidity, rainfall) via Open-Meteo API to adjust degradation curves in real-time.
- **Safety Car Scenarios**: Models the impact of SC/VSC periods on tire aging and strategic "cheap" pit stops.

### 5. Interactive Dashboard
A premium, "Carbon Black" and "F1 Red" themed UI built for rapid strategy visualization:
- Real-time line charts for tire degradation curves (via Recharts).
- Dynamic stint-based strategy bars for comparing alternative race plans.

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
1. Open the UI at `http://localhost:5173`.
2. Select your **Track**, **Year**, and **Driver**.
3. Toggle **Safety Car** scenarios to see how windows shift.
4. Compare **Alternative Strategies** (e.g., Soft-Medium-Medium vs. Soft-Hard).

---

## 🔧 Automation
The project includes a weekly retraining script for the 2026+ era to incorporate new data as it becomes available:
```bash
# Install cron job (Every Monday at 03:00)
(crontab -l 2>/dev/null; echo "0 3 * * 1 cd \"$PWD\" && ./scripts/retrain_active_aero_weekly.sh >> active_aero_retrain.log 2>&1") | crontab -
```

