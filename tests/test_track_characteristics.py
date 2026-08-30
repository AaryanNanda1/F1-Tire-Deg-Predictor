import unittest

from mappings import (
    TRACK_CONFIG,
    get_legacy_track_feature_aliases,
    get_track_characteristic_source,
    get_track_features,
)
from track_characteristics import (
    TRACK_CHARACTERISTICS,
    TRACK_CHARACTERISTIC_SOURCES,
    TRACK_FEATURE_NAMES,
    compute_corner_speed_energy,
    normalize_pirelli_rating,
    parse_turn_speed_pairs,
)


class TrackCharacteristicDerivationTest(unittest.TestCase):
    def test_pirelli_normalization_maps_scale_endpoints_and_midpoint(self):
        self.assertEqual(normalize_pirelli_rating(1), 0.0)
        self.assertEqual(normalize_pirelli_rating(3), 0.5)
        self.assertEqual(normalize_pirelli_rating(5), 1.0)

    def test_pirelli_normalization_rejects_out_of_range_ratings(self):
        with self.assertRaises(ValueError):
            normalize_pirelli_rating(0)
        with self.assertRaises(ValueError):
            normalize_pirelli_rating(6)

    def test_corner_speed_energy_uses_mean_squared_capped_speed(self):
        # 320 is capped to the fixed 300 km/h reference.
        expected = (0.0 + 0.25 + 1.0 + 1.0) / 4
        self.assertAlmostEqual(
            compute_corner_speed_energy([0, 150, 300, 320]),
            expected,
        )

    def test_turn_speed_parser_preserves_turn_numbers(self):
        self.assertEqual(
            parse_turn_speed_pairs("1:95;2:120;10:280"),
            {1: 95, 2: 120, 10: 280},
        )

    def test_catalogue_has_26_source_rows_plus_barcelona_alias(self):
        self.assertEqual(len(TRACK_CHARACTERISTIC_SOURCES), 27)
        self.assertEqual(len(TRACK_CHARACTERISTICS), 27)
        self.assertEqual(
            TRACK_CHARACTERISTICS["Circuit de Barcelona-Catalunya (Spain)"],
            TRACK_CHARACTERISTICS[
                "Circuit de Barcelona-Catalunya (Barcelona, Spain)"
            ],
        )

    def test_bahrain_values_are_derived_from_published_ratings(self):
        features = get_track_features("Bahrain Grand Prix")
        self.assertEqual(
            set(features),
            set(TRACK_FEATURE_NAMES),
        )
        self.assertEqual(features["traction"], 0.75)  # Pirelli 4/5
        self.assertEqual(features["tyre_stress"], 0.5)  # Pirelli 3/5
        self.assertEqual(features["abrasiveness"], 1.0)  # Pirelli 5/5

        source = get_track_characteristic_source("Bahrain Grand Prix")
        self.assertEqual(source["pirelli_ratings"]["abrasiveness"], 5)
        self.assertEqual(source["mercedes_asset_id"], "M496054")

    def test_all_catalogue_values_are_bounded(self):
        for circuit, features in TRACK_CHARACTERISTICS.items():
            with self.subTest(circuit=circuit):
                self.assertEqual(set(features), set(TRACK_FEATURE_NAMES))
                for value in features.values():
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_cota_turn_one_uses_documented_2024_supplement(self):
        source = get_track_characteristic_source("United States Grand Prix")
        self.assertEqual(source["mercedes_missing_turns"], ())
        self.assertEqual(source["mercedes_turn_speeds_kmh"][1], 85)
        self.assertEqual(len(source["mercedes_turn_speeds_kmh"]), 20)
        self.assertEqual(source["mercedes_supplemental_source_year"], 2024)
        self.assertEqual(source["mercedes_supplemental_asset_id"], "M464050")
        self.assertEqual(source["mercedes_supplemental_turns"], (1,))

    def test_historical_rows_replace_paul_ricard_and_mugello_fallbacks(self):
        paul = get_track_features("French Grand Prix")
        paul_source = get_track_characteristic_source("French Grand Prix")
        self.assertEqual(paul_source["reference_season"], 2022)
        self.assertEqual(paul_source["mercedes_asset_id"], "M325101")
        self.assertEqual(paul["traction"], 0.75)
        self.assertEqual(paul["braking_severity"], 0.25)
        self.assertAlmostEqual(paul["corner_speed_energy"], 0.293518518519)

        mugello = get_track_features("Tuscan Grand Prix")
        mugello_source = get_track_characteristic_source("Tuscan Grand Prix")
        self.assertEqual(mugello_source["reference_season"], 2020)
        self.assertEqual(mugello_source["mercedes_asset_id"], "M242537")
        self.assertEqual(mugello["traction"], 0.0)
        self.assertEqual(mugello["lateral_load"], 1.0)
        self.assertAlmostEqual(mugello["corner_speed_energy"], 0.509814814815)

    def test_madrid_and_unknown_circuits_use_neutral_fallback(self):
        madrid = "MADRING (Madrid, Spain)"
        self.assertIn(madrid, TRACK_CONFIG)
        for circuit in (madrid, "Unknown Test Circuit"):
            with self.subTest(circuit=circuit):
                self.assertEqual(
                    get_track_features(circuit),
                    {feature: 0.5 for feature in TRACK_FEATURE_NAMES},
                )
                self.assertIsNone(get_track_characteristic_source(circuit))

    def test_legacy_projection_is_direct_and_has_no_weighted_composites(self):
        features = get_track_features("Monaco Grand Prix")
        legacy = get_legacy_track_feature_aliases(features)
        self.assertEqual(legacy["high_speed_load"], features["corner_speed_energy"])
        self.assertEqual(legacy["surface_roughness"], features["asphalt_grip"])
        self.assertEqual(legacy["thermal_stress"], features["tyre_stress"])
        self.assertEqual(legacy["surface_wear"], features["abrasiveness"])


if __name__ == "__main__":
    unittest.main()
