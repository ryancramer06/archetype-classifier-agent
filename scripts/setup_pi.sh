#!/usr/bin/env bash
# Optional dependency setup. The default portable demo runs with python3 directly.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python3 --version
mode="${1:---lite}"
if [[ "$mode" == "--lite" ]]; then
    python3 scripts/pi_check.py
elif [[ "$mode" == "--core" || "$mode" == "--full" ]]; then
    python3 -m venv .venv
    requirements_file="requirements-pi.txt"
    if [[ "$mode" == "--full" ]]; then requirements_file="requirements.txt"; fi
    if [[ "$mode" == "--core" ]] && compgen -G 'pi_wheels/*.whl' > /dev/null; then
        python3 scripts/verify_wheels.py
        .venv/bin/python -m pip install --no-index --find-links pi_wheels -r "$requirements_file"
    else
        # Git clones omit vendor wheels. Fetch the pinned packages when no cache exists.
        .venv/bin/python -m pip install --only-binary=:all: -r "$requirements_file"
    fi
    .venv/bin/python scripts/pi_check.py --with-sklearn
else
    printf 'Usage: bash scripts/setup_pi.sh [--lite|--core|--full]\n' >&2
    exit 2
fi
