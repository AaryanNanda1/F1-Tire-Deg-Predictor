# F1 Tire Degradation Analysis

 Analysis of Formula 1 tire degradation using Python, pandas, and scikit-learn.

## Setup

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd f1-tire-deg
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Era Model Training

Train two separate era models:
- Ground effect era: `2022-2025`
- Active aero era: `2026-2030` (uses races completed up to `--as-of-date`)

```bash
python3 train_era_models.py --mode both --as-of-date 2026-02-20 --output-dir models
```

Outputs:
- `models/ground_effect_2022_2025_model.joblib`
- `models/ground_effect_2022_2025_features.joblib`
- `models/active_aero_2026_2030_model.joblib` (if active-aero race data exists)
- `models/active_aero_2026_2030_features.joblib`
- `models/era_training_metadata.json`

Weekly active-aero retraining command:

```bash
./scripts/retrain_active_aero_weekly.sh
```

Install weekly cron job (every Monday at 03:00):

```bash
(crontab -l 2>/dev/null; echo "0 3 * * 1 cd \"$PWD\" && ./scripts/retrain_active_aero_weekly.sh >> active_aero_retrain.log 2>&1") | crontab -
```

### Phase-Based Strategy Planner

Use the new phased planner to:
- build weighted pre-race history (previous 3 races + same race last year),
- estimate tire windows for all compounds (including `INTERMEDIATE` and `WET`),
- optimize pre-race strategy and report overstay lap delta.

```bash
python3 run_strategy.py \
  --year 2024 \
  --gp Bahrain \
  --driver VER \
  --team "Red Bull Racing" \
  --race-laps 57 \
  --condition auto \
  --output-json strategy_output.json
```

Key outputs:
- `phase_2_compound_models`: compound lifespan windows and degradation rates
- `phase_3_best_strategies`: top candidate stint plans
- `phase_3_overstay_delta`: lap delta increase when extending a stint beyond window

### Combined Lap + Strategy Prediction

You can run lap-time prediction and strategy planning together from `predict.py`:

```bash
python3 predict.py \
  --with-strategy \
  --year 2024 \
  --gp Bahrain \
  --driver VER \
  --team "Red Bull Racing" \
  --race-laps 57 \
  --lap-number 15 \
  --tyre-life-laps 5 \
  --compound SOFT \
  --stint 1 \
  --condition auto \
  --output-json combined_output.json
```
