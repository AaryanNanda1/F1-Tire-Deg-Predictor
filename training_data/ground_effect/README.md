# Ground Effect Processed Training Data

This directory is the repository-versioned source dataset for reproducible Ground Effect model training.

- `manifest.json` records every session, checksum, row count, schema fingerprint, and relative file path.
- `ground_effect/` contains all eligible Race, Sprint, and FP2 sessions from 2022–2025.
- The coverage block in the manifest confirms that all 187 expected sessions across 92 races are present.
- The compressed session files contain 105,166 processed training rows.

The Ground Effect era is complete and is not modified by the weekly Active Aero workflow. Raw FastF1 responses remain in local or GitHub Actions cache storage and are not committed.

Do not edit compressed session files manually. To retrain the original single model reproducibly, run:

```bash
python train_era_models.py \
  --mode ground_effect \
  --processed-data-dir training_data/ground_effect
```
