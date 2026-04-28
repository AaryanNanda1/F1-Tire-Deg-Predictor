# 🏎️ F1 Tire Strategy & Degradation Predictor

![F1 Tire Strategy Dashboard](/Users/aaryannanda/.gemini/antigravity/brain/82f7d8d7-b465-43a6-a1fd-0db6b6350798/f1_dashboard_screenshot_1777353609861.png)

A high-performance, data-driven analysis engine and dashboard for predicting Formula 1 tire degradation and optimizing race strategies. This project combines machine learning with physical simulation to provide insights across different F1 technical eras.

---

## 🌟 Core Features

### 1. Dual-Output Tire Life Engine
Unlike traditional heuristic models, our engine produces two independent, high-fidelity metrics:
- **Performance Cliff Detection**: Identifies the exact lap where a tire's physics begins to fail. Uses a *Sustained Acceleration Rule* (slope and curvature analysis) to detect when lap times begin to worsen uncontrollably.
- **Strategic Useful Life**: Calculates the "crossover point" where the cumulative time lost to degradation exceeds the time lost in a pit stop.

### 2. Multi-Era Modeling
Supports disparate aerodynamic and tire regulations across F1 history and future:
- **Ground Effect Era (2022–2025)**: Tuned for the current high-downforce technical regulations.
- **Active Aero Era (2026–2030)**: Pre-emptive modeling for the upcoming technical overhaul, including weekly retraining capabilities.

### 3. Advanced Training Methodology
Built for robust "in-season" performance and backtesting:
- **Chronological Filtering (`--as-of-date`)**: Simulates a specific point in time by only training on races completed before the target date. This is critical for evaluating how the model would have performed mid-season.
- **Walk-Forward Validation**: An iterative evaluation loop that trains on previous events and tests on the "next" race, simulating real-world predictive requirements.
- **Compound-Age Weighting**: Specifically increases the training weight of Soft tires as they age (up to 3.0x) to ensure the model accurately captures the performance "cliff."
- **Session-Weighted Learning**: Dynamically weights data based on session type (Race=1.0, Sprint=0.75, FP2=0.5) to prioritize the most representative long-run data.

### 4. Real-Time Strategy Optimization
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

