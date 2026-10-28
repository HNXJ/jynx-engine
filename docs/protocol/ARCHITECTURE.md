# Architecture (logical 1+6)

Jynx is an open-world scientific discovery game built from seven cooperating helpers. Evidence is **immutable** and **content-addressed**. World state is a **reproducible projection** of ledger events plus adjudication outcomes—not a mutable scratch pad.

## Components

- **ARENA** — presents moves, streams events, talks to a versioned API.
- **RULES** — defines legal moves, protocol versions, and governance.
- **SCIENCE** — encodes claims/targets with provenance, uncertainty, contradictions, and revision history.
- **LAB** — analysis tooling, models, and harnesses.
- **ENGINE** — executes simulations and agents.
- **JUDGE** — independent, pluggable validators with typed outcomes and domain rubrics.
- **WORLD** — append-only event ledger; projections are derived.

Helpers version independently with an explicit **dependency DAG** (no cycles).

## Rounds

Round **R_k** is pinned at start: exact release (commit SHA + content digest) per component, compatibility constraints, evaluation input digests, environment identifiers, and active objectives. Once started, the manifest is **immutable**. “Latest release” is only for discovering candidates before a round opens.

## Scoring

Scores are a vector **S = (V, I, C)** (validity, information gain, cost)—never collapsed to a single scalar in protocol artifacts.

## Moves (generalized)

| Move type | Intent |
|-----------|--------|
| `growth_n_to_n_plus_1` | Add one justified unit to the substrate |
| `prediction` | Registered forecast with falsifiable criteria |
| `falsification` | Challenge via counter-evidence (adds events; never erases) |
| `replication` | Independent reproduction |
| `compression` | Loss-bounded summary with stated bounds |
| `experimental_design` | Protocol for reducing ambiguity |
| `uncertainty_reduction` | Quantified uncertainty decrease |

Each move references **acceptance criteria** (`LogicalRef`) and one or more **judges**.

## Roles & identity

Roles: **proposer**, **challenger**, **replicator**, **adjudicator**. Actor identity is persistent and **separate** from model, runtime, or provider.

## Judgments

A Judge **PASS** means the submission satisfied a **rubric under a protocol version**, not that a hypothesis is true. Accepted claims remain challengeable; falsification appends ledger events.

## Tournaments

Tournaments are **bring-your-own** packages under `tournaments/<id>/` with `tournament.yaml`. Rules, engine, judge, and arena contracts are referenced by **logical id + version range**—never repository names, absolute paths, or a specific engine binary.

See `tournaments/neuro-omission-growth/` for the first N→N+1 omission / prediction-error example (one primary objective, one judge; schema supports many).

## Validation before execution

1. JSON Schema validation  
2. Cross-reference resolution against `plugin_registry`  
3. Version compatibility between `protocol_version` and contract ranges  
4. Missing plugins  
5. Prohibited dependency cycles  
6. Held-out evidence availability (sealed registry; see [HELD_OUT_EVIDENCE.md](HELD_OUT_EVIDENCE.md))

Implementation: `src/jynx_protocol/tournament_validator.py`.
