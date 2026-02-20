#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

AS_OF_DATE="$(date +%F)"
python3 train_era_models.py --mode active_aero --as-of-date "${AS_OF_DATE}" --output-dir models
