#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

# Сначала native Trace и сценарии восстановления; при ошибке API не вызывается.
.venv/bin/python -m pytest -q \
    tests/unit/agent/test_loop.py \
    tests/unit/agent/test_finish_recovery.py \
    tests/unit/agent/test_uncapped_requests.py \
    tests/unit/search/test_wikipedia.py \
    tests/unit/search/test_hydra_pilot.py

exec .venv/bin/python examples/run_hydra_pilot10.py --billing internal "$@"
