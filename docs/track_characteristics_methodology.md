# Track-characteristic methodology

## Scope

The circuit priors are a reproducible **latest-available official reference
catalogue**, not hand-tuned estimates and not telemetry-derived tire forces.
The 2025 season is the default reference; the most recent Formula 1 event data
is used for configured circuits that were not on that calendar. The raw
inputs, source URLs, and every Mercedes corner-speed observation are stored in
[`data/track_characteristics.csv`](../data/track_characteristics.csv).
The deterministic implementation is in
[`track_characteristics.py`](../track_characteristics.py).

The [Pirelli Formula 1 press archive](https://press.pirelli.com/?h=1&t=Formula%201)
is the correct archive/discovery source for the Pirelli data. Each catalogue
row links to the specific event-preview article and its Track Characteristics
graphic; the archive page alone is not used as the citation for an individual
rating. Most corner-speed observations come from the
[Mercedes-AMG F1 2025 Track Map search](https://media.mercedesamgf1.com/marsF1/searchresult/searchresult.xhtml?searchString=Track+Map+2025&searchType=detailed),
with the exact asset page and source year recorded per circuit. Historical
rows and supplemental observations link to their exact Mercedes asset pages.

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
- The 2025 Mercedes United States map does not print a minimum-speed value for
  Turn 1. The unchanged-layout 2024 Mercedes map prints 85 km/h for that turn,
  so only Turn 1 is supplemented from asset `M464050`. The primary and
  supplemental years, asset IDs, URLs, and affected turns are stored
  separately in the CSV.
- Paul Ricard uses its most recent Formula 1 event data: Pirelli's 2022 French
  Grand Prix ratings and Mercedes 2022 asset `M325101` for all 15 turn speeds.
- Mugello uses its Formula 1 event data: Pirelli's 2020 Tuscan Grand Prix
  ratings and Mercedes 2020 asset `M242537` for all 15 turn speeds.
- Madrid has no previous Formula 1 event or prior official Pirelli and Mercedes
  event maps for the planned circuit. It therefore remains an explicit neutral
  `0.5` fallback, with no source metadata.
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
- A single reference row is applied to all model years for each circuit. Most
  rows use 2025, while Paul Ricard and Mugello use their latest Formula 1 event
  seasons. This is useful as a stable circuit prior, but it does not capture
  annual resurfacing, setup, weather, or tire-construction changes.
- These features provide context to the learned degradation model; they should
  not be interpreted as standalone tire-life predictions.

## Reproduction checks

Run:

```bash
python -m unittest tests.test_track_characteristics
```

The tests validate the normalization endpoints, Mercedes formula, source-row
coverage, historical and supplemental source metadata, neutral fallbacks,
aliases, and `[0, 1]` bounds.
