# Jynx protocol (1+6 open-world scientific discovery)

Minimal overview of the **Jynx** game protocol: versioned contracts, immutable evidence, independent judges, and portable tournaments. Runtime code in this repository is unchanged by the protocol layer; see `docs/protocol/` for details.

## Seven logical repositories

| Role | Purpose | Current repo (proposal → future name) |
|------|---------|----------------------------------------|
| **ARENA** | Frontend client (API + event stream) | `gamma-arena` → **jynx-arena** (proposal) |
| **RULES** | Legal moves, contracts, governance | `gamma-protocol` → **jynx-rules** |
| **SCIENCE** | Claims, targets, provenance | `gamma-science` → **jynx-science** |
| **LAB** | Analysis, tools, models | `gamma-analysis` + `gamma-tools` → **jynx-lab** |
| **ENGINE** | Execution, agents, simulation | this repo’s runtime → **jynx-engine** (proposal) |
| **JUDGE** | Pluggable validators & rubrics | `gamma-labyrinth` judge → **jynx-judge** |
| **WORLD** | Event ledger & projections | world ledger → **jynx-world** (proposal) |

This checkout (**HNXJ/jynx-arena**) hosts a **provisional** copy of JSON Schemas and validators until they move to **jynx-rules**.

## Documentation

- [Architecture & moves](docs/protocol/ARCHITECTURE.md)
- [Shared repository root layout](docs/protocol/SHARED_ROOT_LAYOUT.md)
- [Schema authority](docs/protocol/SCHEMA_AUTHORITY.md)
- [Held-out evidence](docs/protocol/HELD_OUT_EVIDENCE.md)
- [Replay & reproducibility contract](docs/protocol/REPLAY_REPRODUCIBILITY.md)

## Quick checks

```bash
python3 scripts/check_shared_root.py .
PYTHONPATH=src python3 -m pytest tests/test_protocol_schemas.py tests/test_tournament_validator.py tests/test_shared_root_checker.py -q
```

Example tournament: `tournaments/neuro-omission-growth/tournament.yaml`.
