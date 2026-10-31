# Repository Guidelines

## Project Structure & Module Organization

This project builds an open-source RL environment for search agents and demonstrates training improvements on held-out tasks. It currently contains documentation only:

- `README.md`: scope, infrastructure context, and initial model plan.
- `docs/ENVIRONMENT.md`: observations, actions, rewards, and episode logging.
- `docs/EXPERIMENTS.md`: datasets, evaluation protocol, and training stages.
- `docs/TASKS.md`: agreed ownership and integration boundaries.
- `docs/RELATED_WORK.md`: primary research sources.

Source, tests, datasets, and assets directories have not been created. The sibling `../os_jev_ru` contains legacy reference material; review dependencies before reusing its code.

## Build, Test, and Development Commands

No build system, dependency manifest, executable entry point, or test runner is configured. Use `rg --files .` to inspect files and `rg -n 'pattern' README.md docs/` to search documentation. When adding runnable code, document exact installation, execution, and test commands in `README.md`.

## Coding Style & Naming Conventions

Write project documentation and experiment notes in Russian; use English code identifiers. For new Python code, use four-space indentation, `snake_case` functions and modules, and `PascalCase` classes. Keep Markdown sections descriptive and examples accurate. No formatter or linter is configured; document tooling when introduced.

## Testing Guidelines

No test framework or coverage threshold exists yet. Prefer pytest for new Python tests, with files such as `tests/test_metrics.py` and functions named `test_<behavior>`. Cover reward calculation, duplicate document IDs, budget exhaustion, invalid actions, and deterministic replay. Keep default tests independent of model downloads, paid APIs, and GPUs. Document the test command once configured.

## Commit & Pull Request Guidelines

Git is not initialized, so no historical commit convention exists. Use concise imperative subjects, for example `Add episode result validation`. PRs should explain the change, relevant task, validation performed, and experiment impact. Update affected specifications alongside interface or reward changes.

## Architecture & Experiment Rules

Sergey owns agents, harness, and models; Dima owns corpus storage and search APIs; Kolya owns benchmark execution infrastructure; Vova owns tasks, metrics, and judges. Coordinate shared contracts through `docs/TASKS.md`.

Keep evaluation questions and labels out of training. Record dataset/model revisions, seeds, prompts, budgets, trajectories, and costs in MLflow. Keep credentials outside tracked files. Obtain explicit authorization for paid runs and avoid heavy local computation or model downloads on the Mac.
