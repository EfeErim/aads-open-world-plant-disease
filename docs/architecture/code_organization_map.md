# Repo-Wide Code Organization Map

This map defines where code should live as the project grows. The goal is to keep the repo organized like a shared product platform: common parts live once, while notebooks, scripts, and CLIs remain thin usage surfaces.

## Operating Model

- `src/` is the maintained application platform. Durable business logic, typed contracts, workflows, runtimes, model services, dataset services, and shared utilities live here.
- `scripts/` is the operational surface. Scripts may parse arguments, assemble inputs, call `src/`, and write reports, but reusable logic should move into `src/` once more than one surface needs it.
- `scripts/notebook_cells/` contains notebook cell orchestration. Cells should stay small and delegate real work to `scripts/notebook_helpers/` or `src/`.
- `scripts/notebook_helpers/` contains notebook-specific helpers that are still testable Python modules. Helpers can adapt notebook state, format display output, and call canonical workflows.
- `colab_notebooks/` contains user-facing notebooks. Notebooks should collect parameters, call maintained helpers/workflows, and render results; they should not become independent implementations.
- `tests/` mirrors the maintained surfaces. Unit tests cover helpers and services; integration tests cover canonical workflow and runtime contracts.
- `config/` owns behavior knobs. Policy should be configured here rather than hard-coded into notebooks or one-off scripts.
- `docs/` owns user and maintainer explanations. Durable architectural decisions should be documented here and summarized in `PROJECT_STATE.md` when they affect future work.

## Module Categories

Every Python file should fit one primary category:

| Category | Primary location | Responsibility |
|---|---|---|
| `core` | `src/core/` | Config loading, migrations, environment normalization |
| `domain` | `src/data/`, `src/ood/`, `src/router/`, `src/adapter/` | Focused domain logic with clear inputs and outputs |
| `service` | `src/training/services/` and similar packages | Coordinate domain logic without becoming public entrypoints |
| `workflow` | `src/workflows/` | Stable app-facing facade for training and inference |
| `runtime` | `src/pipeline/`, router runtime modules | Stateful resource loading, adapter discovery, payload assembly |
| `shared` | `src/shared/` | Cross-cutting contracts, path helpers, JSON/CSV/hash/telemetry utilities |

Dataset release ownership is split by lifecycle without duplicating transport code: `src/data/dataset_release.py`
owns local snapshot/audit/quarantine/shard-plan contracts, `src/data/dataset_release_github.py` owns the remote immutable
release lifecycle, and both reuse the GitHub REST/authentication client in `src/pipeline/adapter_release.py`.
| `cli` | `src/app/`, top-level `scripts/*.py` | Argument parsing and command orchestration |
| `notebook_cell` | `scripts/notebook_cells/` | Thin cell-level orchestration |
| `notebook_helper` | `scripts/notebook_helpers/` | Testable notebook adapters around canonical logic |
| `validation` | `scripts/validate_*.py`, `scripts/check_*.py`, `scripts/monitor_*.py` | Repo, artifact, dataset, and config guardrails |
| `test` | `tests/` | Behavioral evidence for the categories above |

## Dependency Direction

Allowed direction:

```text
notebooks -> notebook_cells -> notebook_helpers -> scripts/src
scripts -> src
src/app -> src/workflows
src/workflows -> src services/runtimes/domain/shared
src services/runtimes/domain -> src/shared
tests -> any maintained surface under test
```

Forbidden direction:

- `src/` must not import `scripts`, `tests`, or `colab_notebooks`.
- `src/shared/` must not depend on domain-specific runtime modules.
- Notebook cell files must not become canonical sources of training, routing, OOD, or inference behavior.
- Generated surfaces such as `runs/`, `models/adapters/`, `outputs/`, and `.runtime_tmp/` must not be treated as maintained source.

## Notebook Pattern

Each maintained notebook should follow this shape:

1. bootstrap and access checks
2. parameter and dataset/artifact resolution
3. canonical helper or workflow call
4. result display
5. report/export/publish step

When a notebook needs new behavior, first decide whether it is:

- user interface glue: keep it in a notebook cell/helper
- reusable workflow behavior: move it into `src/workflows/` or a service
- artifact validation: put it under `scripts/validate_*.py`
- experiment-only evaluation: keep it in a script/helper with docs that mark it as research-only

## Refactor Rules

- If the same behavior appears in two notebooks or scripts, extract a helper.
- If the same helper is useful outside notebooks, move it into `src/`.
- If a script exceeds simple orchestration and needs unit tests for internal logic, extract the internal logic into functions or `src/`.
- Keep public workflow signatures stable unless a migration is explicitly planned.
- Prefer additive guardrails before broad file moves.

## Automated Guard

Run the organization audit before large notebook/script refactors:

```powershell
.\scripts\python.cmd scripts/audit_code_organization.py
```

The audit checks import direction and reports a categorized inventory of Python and notebook surfaces. It is intentionally conservative: hard failures are limited to boundary violations, while size and notebook-shape signals are warnings for follow-up refactors.
