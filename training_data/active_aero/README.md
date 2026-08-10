# Active Aero Processed Training Data

This directory is the repository-versioned source dataset for automated Active Aero model retraining.

- `manifest.json` records each session key, role, row count, schema fingerprint, file path, and SHA-256 checksum.
- `active_aero/` contains processed 2026+ Race, Sprint, and FP2 sessions.
- `physics_prior/` contains the frozen 2024–2025 prior used by the current hybrid model.
- Each session is stored independently as a compressed CSV so weekly updates add new files instead of rewriting one large binary dataset.

The current store contains:

- 22 Active Aero sessions and 13,323 processed rows through the 2026 Hungarian Grand Prix.
- 96 frozen prior sessions and 55,956 processed rows covering all of 2024–2025.
- The original single-model training policy uses a deterministic 50% row sample from that prior.

The prior is intentionally frozen. Weekly automation appends only newly completed Active Aero sessions. Raw FastF1 response files are stored separately in GitHub Actions cache storage and are not committed to the repository.

Do not edit the compressed session files manually. Use:

```bash
python scripts/update_active_aero_training_data.py \
  --store-dir training_data/active_aero
```

The original single-model trainer validates the manifest and loads the complete stored dataset with:

```bash
python train_era_models.py \
  --mode active_aero \
  --processed-data-dir training_data/active_aero
```
