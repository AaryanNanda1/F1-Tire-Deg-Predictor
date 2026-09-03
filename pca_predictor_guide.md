# Predictor, PCA, and Heat-Map Guide

This document describes the predictors used by the current single-model tire
degradation engine and a practical workflow for investigating them with PCA
and correlation heat maps.

The Ground Effect and Active Aero models currently use the same 37 raw
predictors. Categorical columns are stored as raw values in processed
sessions. Each trained model is a scikit-learn `Pipeline` containing a fitted
`ColumnTransformer`, imputation, one-hot encoding, and the
`HistGradientBoostingRegressor`.

## Current predictors

### Driver and session context

| Predictor | Type | Meaning |
|---|---|---|
| `Driver` | categorical | Driver identifier, such as `VER` or `HAM`. |
| `Team` | categorical | Normalized team identifier. |
| `Compound` | categorical | Tire compound: `SOFT`, `MEDIUM`, `HARD`, `INTERMEDIATE`, or `WET`. |
| `SessionCode` | categorical | `R` for Race, `S` for Sprint, or `FP2` for Practice 2. |
| `TrackType` | categorical | Sourced track classification: `Slow`, `Medium`, or `Fast`. |
| `Stint` | numeric/discrete | FastF1 stint number for the driver in that session. |
| `IsWet` | binary | Indicates wet/intermediate conditions or a wet compound. |

### Tire age and race position

| Predictor | Meaning |
|---|---|
| `LapNumber` | Session lap number. In simulation it is held at a fixed race-context lap to isolate tire age. |
| `TyreLife` | FastF1 tire age in laps. |
| `TyreLifeKM` | Tire age multiplied by estimated circuit length. |
| `TyreLifeSquared` | Nonlinear tire-age term for late-stint degradation. |
| `NormalizedLap` | `LapNumber / race_laps`, bounded to 0–1. |
| `LapsRemaining` | Estimated race laps remaining. |
| `TireAgeRatio` | `TyreLife / race_laps`, bounded to 0–1. |
| `FuelLoad` | Lap-number fuel proxy for Race/Sprint sessions. FP2 receives a neutral sentinel because FP2 lap number does not identify fuel load. |
| `FuelLoadMissing` | Missing-fuel indicator; `1` for FP2 and `0` for Race/Sprint. |

### Weather

| Predictor | Meaning |
|---|---|
| `AirTemp` | Air temperature from FastF1 weather data. |
| `TrackTemp` | Track temperature from FastF1 weather data. |
| `Humidity` | Relative humidity from FastF1 weather data. |
| `Rainfall` | Rainfall indicator/value from FastF1 weather data. |
| `WindSpeed` | Wind speed from FastF1 weather data. |

Missing numeric weather values are preserved during preprocessing. The fitted
training pipeline uses a median imputer and adds missingness indicators before
model fitting.

### Sourced track characteristics

These are circuit-level variables derived from the track-characteristics
catalogue and Pirelli/source-backed ratings. They are constant for all rows at
the same circuit.

| Predictor | Meaning |
|---|---|
| `traction` | Traction demand. |
| `tyre_stress` | Overall tire stress. |
| `asphalt_grip` | Asphalt grip level. |
| `corner_speed_energy` | Corner-speed energy demand. |
| `abrasiveness` | Surface abrasiveness. |
| `braking_severity` | Braking severity. |
| `lateral_load` | Lateral tire load. |

### Engineered interactions

| Predictor | Meaning |
|---|---|
| `tire_age_x_abrasiveness` | Tire age multiplied by abrasiveness. |
| `track_temp_x_tyre_stress` | Track temperature multiplied by tire stress. |
| `tire_age_x_traction` | Tire age multiplied by traction demand. |
| `tire_age_x_lateral_load` | Tire age multiplied by lateral load. |
| `soft_age_interaction` | Tire age when the compound is Soft; otherwise zero. |
| `medium_age_interaction` | Tire age when the compound is Medium; otherwise zero. |
| `hard_age_interaction` | Tire age when the compound is Hard; otherwise zero. |
| `soft_abrasiveness_interaction` | Soft-compound indicator multiplied by abrasiveness. |
| `soft_traction_interaction` | Soft-compound indicator multiplied by traction demand. |

## Columns intentionally excluded

`LapTimeDelta` is the target, not a predictor. `EventDate`, `EventName`,
`SessionKey`, `TrainingRole`, and `Season` are provenance/evaluation metadata.
They must not be included in PCA or model inputs when studying generalization.

The following leakage-prone or legacy fields must also remain excluded:

- `StintLength`
- `NormalizedTyreLife`
- `normalized_life_x_tyre_stress`
- `TeamBaselinePace`
- `FieldBaselinePace`
- `RelativePace`
- Per-session dummy columns such as `EventName_*`, `Driver_*`, or `Compound_*`

## What PCA answers

PCA identifies directions of greatest variance in the predictor matrix. It can
show whether predictors are redundant, whether weather and track variables
form separate regimes, and whether the data is dominated by driver/team or
compound categories. PCA is exploratory: a high loading does not prove causal
importance or that removing a variable will improve model MAE.

Because the model contains mixed numeric and categorical data, PCA should be
run after preprocessing. Two useful views are:

1. **Numeric-only PCA**: scale continuous predictors and inspect physical
   relationships without allowing high-cardinality one-hot columns to
   dominate.
2. **Full encoded PCA**: transform the complete raw predictor table through
   the fitted `ColumnTransformer`, scale the resulting columns, and inspect
   the combined numeric/one-hot representation.

For model-development decisions, fit the transformer and scaler on the
training fold only. Transform the held-out event with those fitted objects.
Do not fit PCA on all events before a validation split.

## Creating PCA charts

Use a canonical processed table loaded from a store. Do not independently run
`pd.get_dummies` for each session. A minimal numeric-only example is:

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

numeric = [
    "LapNumber", "TyreLife", "TyreLifeKM", "TyreLifeSquared",
    "FuelLoad", "FuelLoadMissing", "IsWet", "AirTemp", "TrackTemp",
    "Humidity", "Rainfall", "WindSpeed", "traction", "tyre_stress",
    "asphalt_grip", "corner_speed_energy", "abrasiveness",
    "braking_severity", "lateral_load", "NormalizedLap",
    "LapsRemaining", "TireAgeRatio", "tire_age_x_abrasiveness",
    "track_temp_x_tyre_stress", "tire_age_x_traction",
    "tire_age_x_lateral_load", "soft_age_interaction",
    "medium_age_interaction", "hard_age_interaction",
    "soft_abrasiveness_interaction", "soft_traction_interaction",
]

X_numeric = frame[numeric].copy()
X_scaled = StandardScaler().fit_transform(X_numeric)
pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)

chart = pd.DataFrame(components, columns=["PC1", "PC2"], index=frame.index)
chart["Compound"] = frame["Compound"].values
chart["SessionCode"] = frame["SessionCode"].values

for label, group in chart.groupby("Compound"):
    plt.scatter(group["PC1"], group["PC2"], s=8, alpha=0.35, label=label)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend()
plt.tight_layout()
plt.show()
```

Useful PCA charts include:

- PC1 vs PC2 colored by `Compound`.
- PC1 vs PC2 colored by `SessionCode` to check Race/FP2/Sprint separation.
- PC1 vs PC2 colored by `IsWet`.
- A loading bar chart showing the largest positive and negative contributors
  to PC1 and PC2.
- A cumulative explained-variance plot to show how many components summarize
  the data.

For loading interpretation:

```python
loadings = pd.DataFrame(
    pca.components_.T,
    index=numeric,
    columns=["PC1", "PC2"],
)
print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
```

If PCA is used to alter the production model, compare the original and PCA
versions using identical chronological folds and the same target. PCA is
unsupervised, so it may preserve variance unrelated to tire degradation and
may remove low-variance predictors that are useful for rare wet compounds.

## Creating predictor heat maps

For a numeric correlation heat map:

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = frame[numeric + ["LapTimeDelta"]].corr(method="spearman")
plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap="vlag", center=0, square=True)
plt.tight_layout()
plt.show()
```

Spearman correlation is a useful first view because several relationships
with tire age are nonlinear. Pearson correlation can be added as a second
view when linear association is specifically relevant.

Interpret the heat map carefully:

- `TyreLife`, `TyreLifeKM`, `TyreLifeSquared`, and tire-age interactions are
  expected to be strongly correlated. This is structural feature expansion,
  not necessarily a bug.
- Track-characteristic variables are constant within a circuit, so their
  correlations can be driven by the distribution of circuits in the dataset.
- Weather correlations may reflect session timing rather than physics.
- Correlation with `LapTimeDelta` is not feature importance and does not
  replace walk-forward evaluation.

For categorical predictors, use a separate count/association chart rather
than treating arbitrary category labels as numbers. One-hot columns may be
included in a secondary encoded heat map, but interpret them as category
presence indicators and fit the encoder only on the training fold.

## Recommended analysis sequence

1. Load one canonical store and verify its schema version.
2. Separate metadata and target from the 37 raw predictors.
3. Inspect missingness before imputation.
4. Plot numeric distributions and the Spearman heat map.
5. Fit numeric-only PCA with `StandardScaler` on the training fold.
6. Inspect loadings and explained variance, colored by compound/session regime.
7. Repeat with the fitted full `ColumnTransformer` if category effects are of
   interest.
8. Test any feature removal or PCA replacement with chronological,
   event-level holdouts and report MAE by session type and compound.

PCA charts and heat maps are diagnostic tools. The production decision should
be based on held-out event performance, especially for Intermediate and Full
Wet observations where the sample size is very small.
