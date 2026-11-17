# Repository Guidelines

## Current Direction and Project Memory

Read `MEMORY.md` before changing scope. Build the static harness without a Verifiers dependency and measure its baseline first. The primary method is a runtime GEPA-like optimizer of prompts, followed by editable text skills. Compare against static and Hermes-like memory-only. Use docs/RUNTIME_OPTIMIZER.md and docs/EXPERIMENTS.md. Restrict runtime experience to public trajectories and internal critic signals. Keep gold, external scores and future tasks private to evaluation.

## Project Structure

The repository contains a Python package, examples, configs and tests:

- `src/search_rl_env/agent/`, `environment/`, `contracts/`: native Verifiers harness and contracts.
- `src/search_rl_env/search/`: SQL/search services, SDKs and isolated backend adapters.
- `src/search_rl_env/ui/`: local UI.
- Planned `memory/`: persistent notes, session recall, skills, maintenance and snapshots.
- Planned `benchmarks/`, `evaluation/`: datasets, evaluators, stream protocols and experiment runners.
- `docs/PAPER_PLAN.md`: article claim, system and evidence plan.
- `docs/MEMORY_CORE.md`: target memory architecture; it is not an implementation claim.
- `docs/EXPERIMENTS.md`: closed evaluation and chronological online-use protocol.
- `docs/TASKS.md`, `BACKLOG.md`, `ROADMAP.md`: owners, dependencies and milestones.

The sibling `../os_jev_ru` is legacy reference material, not a runtime dependency. Preserve existing uncommitted work.

## Build and Tests

Use the exact installation and execution commands in `README.md`. Python 3.11+ and `pyproject.toml` are configured. Default checks use pytest/pytest-asyncio: `.venv/bin/python -m pytest -q`. Documentation changes require working local links and `git diff --check`.

Keep ordinary tests independent of paid APIs, model downloads and GPUs. Memory checks should cover persistence across sessions/restarts, scope isolation, deterministic event ordering, update/restore, stale records, cancellation and resume. Existing reward, duplicate-id, invalid-action and budget tests remain relevant.

## Style and Workflow

Write documentation and experiment notes in Russian; use English code identifiers. Python uses four-space indentation, snake_case functions/modules and PascalCase classes. Keep executable examples accurate. Follow `CONTRIBUTING.md` for branches, PRs and reviews. Commit subjects are concise and imperative. Update affected specifications with interface changes; do not present planned APIs as implemented commands.

## Ownership and Contracts

Sergey owns the agent, harness and memory policy. Dima owns corpus/SQL storage, search APIs, snapshots and indexes. Kolya owns benchmark execution, event logs, resume, artifacts and costs. Vova owns open/custom benchmark tasks, feedback scenarios, metrics and judges. Coordinate shared interfaces through `docs/TASKS.md` and `docs/CONTRACTS.md`.

Target runtime and evaluator must run without Verifiers. The existing implementation documents the prior prototype. Do not invent training fields in API rollouts. Memory recall and source-document RAG are separate data paths.

## Evaluation and Resources

Closed evaluation uses frozen memory and excludes test records from adaptive recall. Online-use evaluation is chronological: answer first, record the independent score, then provide only the declared feedback and update memory for later tasks. Do not read future tasks or labels or revise previous scores. The two protocols must remain explicitly distinct.

Build a permitted view of native Trace for memory recall. Private TaskData reference answers, document labels and gold SQL must not enter its index directly; only declared feedback may expose corrections after scoring.

Log dataset/model/code revisions, seeds, prompts, event order, budgets, memory snapshots, retrieved records, trajectories and all costs in manifests/MLflow. Keep credentials out of tracked files. Paid API/cloud/GPU runs require explicit budget authorization; avoid heavy local computation and model downloads on the Mac.
