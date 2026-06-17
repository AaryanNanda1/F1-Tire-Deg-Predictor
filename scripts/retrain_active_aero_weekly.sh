#!/bin/bash
set -e

# Navigate to the project directory
cd /Users/aaryannanda/F1-Tire-Deg-Predictor

echo "Starting model retraining at $(date)..."

# Run the training script (specifying active_aero mode to update the 2026+ models)
python train_era_models.py --mode active_aero

# Check if there are changes in the models directory
if git diff --quiet models/; then
    echo "No changes in trained models. Nothing to commit."
else
    echo "New model files detected. Committing and pushing to GitHub..."
    git add models/
    git commit -m "Auto-retrain: update Active Aero model $(date +'%Y-%m-%d')"
    git push origin main
    echo "GitHub repository updated successfully with the new model!"
fi
