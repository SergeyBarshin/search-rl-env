#!/bin/sh
set -eu

mkdir -p /artifacts
python -m pip freeze > /artifacts/packages.txt
python -c 'import faiss, psutil, sqlite3; from importlib.util import find_spec; from importlib.metadata import version; assert find_spec("verifiers") is None; print("Agents SDK", version("openai-agents")); print("FAISS", faiss.__version__, "SQLite", sqlite3.sqlite_version)'

test_status=0
python -m pytest -q --junitxml=/artifacts/pytest.xml > /artifacts/pytest.log 2>&1 || test_status=$?
cat /artifacts/pytest.log

acceptance_status=null
if [ "${SEARCH_RL_SKIP_ACCEPTANCE:-0}" != 1 ]; then
    acceptance_status=0
    python examples/accept_environments.py --output /artifacts/acceptance > /artifacts/acceptance.log 2>&1 || acceptance_status=$?
    cat /artifacts/acceptance.log
fi

printf '{"pytest_exit_code": %s, "acceptance_exit_code": %s}\n' "$test_status" "$acceptance_status" > /artifacts/checks.json
if [ "$test_status" -ne 0 ]; then
    exit 1
fi
if [ "$acceptance_status" != null ] && [ "$acceptance_status" -ne 0 ]; then
    exit 1
fi
