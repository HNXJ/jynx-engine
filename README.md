# jynx-engine

Execution runtime (agents, simulation) and provisional home of the jynx protocol schemas and validators.

| Repo | Role |
|---|---|
| [jynx-arena](https://github.com/HNXJ/jynx-arena) | Frontend |
| [jynx-rules](https://github.com/HNXJ/jynx-rules) | Moves, contracts, governance |
| [jynx-science](https://github.com/HNXJ/jynx-science) | Claims, targets |
| [jynx-lab](https://github.com/HNXJ/jynx-lab) | Analysis, tools |
| [jynx-judge](https://github.com/HNXJ/jynx-judge) | Validators, rubrics |
| [jynx-world](https://github.com/HNXJ/jynx-world) | Event ledger, world state |

- Runtime in `src/`, `config/`, `dashboard/`; protocol in `schemas/`, `src/jynx_protocol/`, `docs/protocol/`.
- Example tournament: `tournaments/neuro-omission-growth/`.

Root layout: `artifacts/ docs/ src/ schemas/ tests/ tournaments/` (`python3 scripts/check_shared_root.py .`).
