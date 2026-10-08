#!/usr/bin/env bash
# Recheck the locally delivered 2026-10-09 release in a separate environment.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
release_dir="${project_dir}/dist/learning-release-20261009"
result_dir="${project_dir}/dist/learning-acceptance-$(date -u +%Y%m%dT%H%M%S%N)"

if [[ ! -f "${release_dir}/install.py" ]]; then
  printf '%s\n' 'Missing local release: see docs/learning_deployment.md for preparation and packaging.' >&2
  exit 1
fi

printf 'Acceptance output: %s\n' "${result_dir}"
exec python3 -I "${release_dir}/install.py" \
  --expected-release-sha256 ea5a1ccfbfc5c8c5d8be17334eecb01fca3ef51626efc9044176e53551f67f42 \
  --venv "${project_dir}/dist/learning-runtime-20261009" \
  --accept-output "${result_dir}" "$@"
