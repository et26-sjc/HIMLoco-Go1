#!/usr/bin/env bash
# Build and transfer one explicit release. No downloads, deletion or startup.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 CONFIG.json POLICY.pt USER@HOST:/absolute/new/release [LOCAL_BUNDLE_DIR]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config=$1
model=$2
destination=$3
python_bin=${PYTHON:-python3}
if [[ $# -eq 4 ]]; then
  bundle=$4
else
  # Keep this reviewable local release after transfer; do not delete user files.
  bundle="$(mktemp -d "${TMPDIR:-/tmp}/himloco-release.XXXXXXXX")/bundle"
fi
"$python_bin" "$script_dir/build_deployment_bundle.py" \
  --config "$config" --model "$model" --output "$bundle"
"$python_bin" "$script_dir/build_deployment_bundle.py" --verify "$bundle"
rsync -av --checksum -e ssh -- "$bundle/" "$destination/"
echo "Transferred bundle; no robot process was started. Local copy: $bundle"
echo "On the destination, verify manifest.json using build_deployment_bundle.py --verify RELEASE_DIR."
echo "Then run deploy_policy.py --config RELEASE_DIR/himloco.json --check on the target runtime."
