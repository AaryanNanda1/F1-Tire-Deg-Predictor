#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

append_summary() {
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        printf "%b\n" "$1" >> "${GITHUB_STEP_SUMMARY}"
    fi
}

on_failure() {
    status=$?
    attempted_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "Active Aero retraining failed with exit code ${status}."
    append_summary "## Active Aero retraining failed\n- Attempted at: \`${attempted_at}\`\n- Exit code: \`${status}\`"
    exit "${status}"
}

trap on_failure ERR

echo "Starting Active Aero model retraining at $(date)..."

# Weekly automation only retrains the Active Aero model for the 2026+ era.
"${PYTHON_BIN}" train_era_models.py --mode active_aero

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    "${PYTHON_BIN}" - <<'PY' >> "${GITHUB_STEP_SUMMARY}"
import json
from pathlib import Path

metadata_path = Path("models/era_training_metadata.json")
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    active = metadata.get("active_aero_2026_2030", {})
    loaded_events = active.get("loaded_events", [])
    failed_events = active.get("failed_events", [])
    print("## Active Aero retraining result")
    print(f"- Status: `{active.get('status', 'unknown')}`")
    print(f"- As-of date: `{active.get('as_of', 'unknown')}`")
    print(f"- Trained at: `{active.get('trained_at', 'unknown')}`")
    print(f"- Loaded sessions: `{len(loaded_events)}`")
    print(f"- Failed race/session loads: `{len(failed_events)}`")
    if failed_events:
        print("- Failed loads:")
        for item in failed_events[:12]:
            print(f"  - `{item}`")
        if len(failed_events) > 12:
            print(f"  - plus {len(failed_events) - 12} more")
else:
    print("## Active Aero retraining result")
    print("- Metadata file was not found after training.")
PY
fi

# Check if there are changes in the models directory
if git diff --quiet -- models/; then
    echo "No changes in trained models. Nothing to commit."
    append_summary "## Active Aero commit result\n- No changes in \`models/\`; nothing was committed."
else
    echo "New model files detected. Committing and pushing to GitHub..."
    if ! git config user.name >/dev/null; then
        git config user.name "github-actions[bot]"
    fi
    if ! git config user.email >/dev/null; then
        git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    fi
    git add models/
    git commit -m "Auto-retrain: update Active Aero model $(date +'%Y-%m-%d')"
    git push origin HEAD:${GITHUB_REF_NAME:-main}
    echo "GitHub repository updated successfully with the new model!"
    append_summary "## Active Aero commit result\n- Model artifacts or metadata changed and were pushed to \`${GITHUB_REF_NAME:-main}\`."
fi
