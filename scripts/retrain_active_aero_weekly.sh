#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-training_data/active_aero}"
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

# Append only newly completed sessions. Historical training rows come from the
# repository dataset; the Actions cache remains a secondary raw-response cache.
"${PYTHON_BIN}" scripts/update_active_aero_training_data.py \
    --store-dir "${PROCESSED_DATA_DIR}" \
    --json-output /tmp/active_aero_data_update.json

# Retrain the original single Active Aero model from the persistent dataset.
"${PYTHON_BIN}" train_era_models.py \
    --mode active_aero \
    --processed-data-dir "${PROCESSED_DATA_DIR}"

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

metadata = json.loads(
    Path("models/era_training_metadata.json").read_text(encoding="utf-8")
)
active = metadata.get("active_aero_2026_2030", {})
if not str(active.get("status", "")).startswith("trained"):
    raise SystemExit("Active Aero retraining did not produce a trained model.")
if active.get("training_data_source") != "persistent_processed_sessions":
    raise SystemExit("Active Aero retraining did not use the persistent dataset.")
if active.get("mae_validation_scope") != "walk_forward_2026_plus_test_events":
    raise SystemExit("Active Aero retraining used the wrong validation scope.")
if not Path("models/active_aero_2026_2030_model.joblib").is_file():
    raise SystemExit("The Active Aero model artifact is missing.")
if not Path("models/active_aero_2026_2030_features.joblib").is_file():
    raise SystemExit("The Active Aero feature manifest is missing.")
print(
    "Validated persistent-data single-model retraining from "
    f"{active.get('active_session_count')} Active Aero sessions."
)
PY

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

# Check if there are model or persistent-data changes.
if [ -z "$(git status --porcelain -- models/ "${PROCESSED_DATA_DIR}/")" ]; then
    echo "No changes in the model or persistent dataset. Nothing to commit."
    append_summary "## Active Aero commit result\n- No model or persistent-data changes; nothing was committed."
else
    echo "New model or processed training data detected. Committing and pushing to GitHub..."
    if ! git config user.name >/dev/null; then
        git config user.name "github-actions[bot]"
    fi
    if ! git config user.email >/dev/null; then
        git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    fi
    git add \
        models/active_aero_2026_2030_model.joblib \
        models/active_aero_2026_2030_features.joblib \
        models/era_training_metadata.json \
        "${PROCESSED_DATA_DIR}/"
    git commit -m "Auto-retrain: update Active Aero model $(date +'%Y-%m-%d')"
    git push origin HEAD:${GITHUB_REF_NAME:-main}
    echo "GitHub repository updated successfully with the new model!"
    append_summary "## Active Aero commit result\n- Model artifacts or metadata changed and were pushed to \`${GITHUB_REF_NAME:-main}\`."

    if [ -n "${RENDER_DEPLOY_HOOK_URL:-}" ]; then
        echo "Triggering Render deploy hook..."
        curl --fail --silent --show-error --request POST "${RENDER_DEPLOY_HOOK_URL}"
        echo "Render deploy hook triggered successfully."
        append_summary "## Render deploy hook\n- Triggered Render deploy hook after pushing the model commit."
    else
        echo "RENDER_DEPLOY_HOOK_URL is not set. Skipping Render deploy hook."
        append_summary "## Render deploy hook\n- Skipped because \`RENDER_DEPLOY_HOOK_URL\` is not configured."
    fi
fi
