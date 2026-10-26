#!/usr/bin/env bash
# Historical filename retained; default behavior is an offline model check.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIG.json [--check|--inspect-config|--observe|--run]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode=${2:---check}
case "$mode" in
  --check|--inspect-config|--observe|--run) ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
exec "${PYTHON:-python3}" "$script_dir/../scripts/deploy_policy.py" --config "$1" "$mode"
