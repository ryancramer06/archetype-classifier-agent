#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python demo.py "$@"
else
    exec python3 demo.py --backend portable "$@"
fi
