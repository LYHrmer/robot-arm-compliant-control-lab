#!/usr/bin/env bash
# Recheck the validated BC input-mask release in a separate environment.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
release_dir="${project_dir}/dist/learning-release-bcfix-20261009"
result_dir="${project_dir}/dist/learning-acceptance-$(date -u +%Y%m%dT%H%M%S%N)"

if [[ ! -f "${release_dir}/install.py" ]]; then
  printf '%s\n' 'Missing local release: see docs/learning_deployment.md for preparation and packaging.' >&2
  exit 1
fi

printf 'Acceptance output: %s\n' "${result_dir}"
exec python3 -I "${release_dir}/install.py" \
  --expected-release-sha256 4e0ba5ea2fff507454f6f8cf705d6f3502f7591ba1b9eb96137a0202dc2df79f \
  --venv "${project_dir}/dist/learning-runtime-bcfix-20261009" \
  --accept-output "${result_dir}" "$@"
