# Shared repository root layout

Every Jynx helper repository exposes **exactly seven required root entries** (additional files and directories are allowed):

| Entry | Kind | Purpose |
|-------|------|---------|
| `artifacts/` | directory | Build outputs, pinned bundles, non-authoritative copies |
| `docs/` | directory | Human-readable specifications |
| `src/` | directory | Implementation source |
| `schemas/` | directory | JSON Schemas and fixtures |
| `tests/` | directory | Automated tests |
| `tournaments/` | directory | Portable tournament packages (`tournaments/<id>/tournament.yaml`) |
| `README.md` | file | Minimal orientation (details live under `docs/`) |

Empty directories may contain a `.gitkeep` file.

## Checker

Run from any repository root:

```bash
python3 scripts/check_shared_root.py /path/to/repo
```

The script exits `0` when all seven entries exist and `1` otherwise, printing each missing path on stderr.

Copy `scripts/check_shared_root.py` into other repos or invoke it from a shared tooling package once **jynx-rules** publishes a release artifact.

## This repository

`HNXJ/jynx-arena` satisfies the layout via `artifacts/`, `docs/`, `src/`, `schemas/`, `tests/`, `tournaments/`, and `README.md`. Legacy paths (for example `config/`, `dashboard/`) remain for the existing runtime and are outside the uniform-root contract.
