# Track-characteristic methodology

## Scope

The circuit priors are a reproducible **2025 reference catalogue**, not
hand-tuned estimates and not telemetry-derived tire forces. The raw inputs,
source URLs, and every Mercedes corner-speed observation are stored in
[`data/track_characteristics_2025.csv`](../data/track_characteristics_2025.csv).
The deterministic implementation is in
[`track_characteristics.py`](../track_characteristics.py).

The [Pirelli Formula 1 press archive](https://press.pirelli.com/?h=1&t=Formula%201)
is the correct archive/discovery source for the Pirelli data. Each catalogue
row links to the specific 2025 event-preview article and its Track
Characteristics graphic; the archive page alone is not used as the citation
for an individual rating. Corner-speed observations come from the
[Mercedes-AMG F1 2025 Track Map search](https://media.mercedesamgf1.com/marsF1/searchresult/searchresult.xhtml?searchString=Track+Map+2025&searchType=detailed),
with the exact asset page recorded per circuit.

## Published inputs

Six features are the integer ratings printed on Pirelli's official Track
Characteristics graphic:

| Model feature | Published Pirelli label | Raw scale |
|---|---|---:|
| `traction` | Traction | 1–5 |
| `tyre_stress` | Tyre Stress | 1–5 |
| `asphalt_grip` | Asphalt Grip | 1–5 |
| `braking_severity` | Braking | 1–5 |
| `abrasiveness` | Asphalt Abrasion | 1–5 |
| `lateral_load` | Lateral | 1–5 |

The seventh feature, `corner_speed_energy`, is derived from the minimum
corner speeds printed beside numbered turns on the Mercedes track map. Gear
annotations are not used.

The previous `surface_roughness` and `track_temp_sensitivity` labels have been
removed because neither quantity is published on these source graphics.
`asphalt_grip` and `tyre_stress` now retain the source's actual terminology.
The former weighted `thermal_stress`, `surface_wear`, and `energy_load`
composites were also removed because their weights did not have an empirical
derivation.

## Normalization

### Pirelli ratings

For a published rating \(r \in \{1,2,3,4,5\}\), the normalized value is:

\[
x = \frac{r - 1}{5 - 1}
\]

This maps 1 to 0.00, 2 to 0.25, 3 to 0.50, 4 to 0.75, and 5 to 1.00. It
preserves the order and equal spacing of Pirelli's scale without introducing
additional weights.

### Mercedes minimum-corner-speed score

For the \(N\) published minimum corner speeds \(v_i\), in km/h:

\[
\text{corner\_speed\_energy} =
\frac{1}{N}\sum_{i=1}^{N}
\left(\frac{\min(v_i, 300)}{300}\right)^2
\]

The fixed 300 km/h reference bounds the score to `[0, 1]`. Squaring is an
energy-like transform that gives high-speed turns more influence than slow
turns; it does **not** claim to calculate actual kinetic energy, lateral force,
or tire load. Values above 300 km/h are capped so a flat-out kink does not
dominate an entire circuit descriptor.

## Missing-data policy

- The 2025 Mercedes search does not contain separate Monaco or Spain maps.
  Their 2024 maps are used because the raced layouts were unchanged; both rows
  carry `mercedes_source_year=2024` and an explanatory note.
- The Mercedes United States map does not print a minimum-speed value for Turn
  1. Turn 1 is recorded in `mercedes_missing_turns`, excluded from \(N\), and
  is not imputed.
- Paul Ricard, Mugello, and Madrid do not have rows in this 2025 catalogue.
  `get_track_features` returns the neutral value `0.5` for all seven features
  rather than preserving unsupported estimates. The API returns no source
  metadata for those fallbacks, and the dashboard labels them as neutral.
- Unknown circuits use the same explicit neutral fallback.

## Training and serving

New training runs use the seven source-native features plus the following
explicit interactions:

- `tire_age_x_abrasiveness`
- `track_temp_x_tyre_stress`
- `tire_age_x_traction`
- `tire_age_x_lateral_load`
- `normalized_life_x_tyre_stress`

The inference engine also supplies direct compatibility aliases when it loads
one of the older committed model schemas. These aliases are a temporary
serving bridge, not part of new training. Models should be retrained before
using the new features for evaluation or production decisions.

## Interpretation and limitations

- Pirelli's ratings are ordinal expert descriptors, not measured continuous
  variables. Linear normalization does not make the gaps physically exact.
- The Mercedes speeds are map annotations and can reflect a representative car
  setup or simulation. They are not observations from every lap or driver.
- A single 2025 reference value is currently applied to all model years. This
  is useful as a stable circuit prior, but it does not capture annual
  resurfacing, layout changes, weather, setup, or tire-construction changes.
- These features provide context to the learned degradation model; they should
  not be interpreted as standalone tire-life predictions.

## Reproduction checks

Run:

```bash
python -m unittest tests.test_track_characteristics
```

The tests validate the normalization endpoints, Mercedes formula, source-row
coverage, missing-turn policy, neutral fallback, aliases, and `[0, 1]` bounds.
