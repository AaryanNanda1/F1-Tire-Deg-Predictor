#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Starting model retraining at $(date)..."

# Run the training script (specifying active_aero mode to update the 2026+ models)
python train_era_models.py --mode active_aero

# Check if there are changes in the models directory
if git diff --quiet -- models/; then
    echo "No changes in trained models. Nothing to commit."
else
    echo "New model files detected. Committing and pushing to GitHub..."
    git add models/
    git commit -m "Auto-retrain: update Active Aero model $(date +'%Y-%m-%d')"
    git push origin HEAD:${GITHUB_REF_NAME:-main}
    echo "GitHub repository updated successfully with the new model!"
fi
