# agents

Execution engine: agents, simulation, protocol schemas and validators.

## Layout

- `src/` packages (`jynx_protocol`, `gamma_runtime`, `sde_engine`, `gamma_peft`, `guard`, `apps`), `src/scripts/check_shared_root.py`.
- `schemas/` JSON schemas. `tournaments/<id>/tournament.yaml`. `tests/` pytest. `docs/` protocol docs.
- `context/` read by the runtime; keep at root. `artifacts/archive/` holds retired files; do not import from it.
- `src/computational/jbiophysic` is a git submodule.

## Build and test

```
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

Gate: the command above. Read its summary line; push only when failures do not grow versus `artifacts/state.md`.

## Git

- Work on `_temp_<name>` branches; PR to `main`. No direct push to `main`.
- Stage exact paths; never `git add .` or `-A`.
- Never commit `.env`, tokens or keys.
