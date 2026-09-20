# GEMINI.md — gamma
Purpose: authoritative runtime/schema/validator/harness repo.
Authority boundary:
- Live truth gates and receipts belong here.

## 1. Execution Infrastructure
- **Python Environment:** `/Users/hamednejat/workspace/gemini-cli-env`
- **Remote Node (Office Mac):** `HN@100.69.184.42`; world root is `/Users/HN/gamma-world`.
- **SSH Doctrine:** Use strictly pinned-key SSH: `ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 HN@100.69.184.42 '<command>'`. Never use plain SSH passwords or clear the SSH agent.
- **State Parity:** Local and remote nodes are a unified state machine. Validate routing remotely before front-end execution.
- **UI/Ports:** Gamma Hub UI is typically `3012`. Verify via HTTP before restarting any live services.
- **Git:** Remote `gamma` MUST be synchronized strictly via git; never use manual `scp`/`rsync`.

## 2. Authorization Scopes
- **Local:** Pre-authorized for all file edits and test executions in the agent's launch directory.
- **Remote:** Pre-authorized strictly within `/Users/HN/MLLM/` and its subfolders. Ask for permission outside this path.

## 3. Council Orchestration & Agentic Roles
- **Gamma Scientific Council:** Agents MUST use code for simulation/analysis, not just token-to-token text work.
- **Hellenic Mapping:** Auditor (Beta), Refiner (Alpha), Executor (Gamma), Inspector (Theta).

## 4. Instructional Chains / Teacher-Student Runtime Conditioning
- **Current Status:** The 8 Gemma slots are treated as `runtime_teacher_prompting` unless verified provenance artifacts exist. Outputs are proposals, not Truth-plane state.

## 5. Gamma Arena / Council Ground Rules
- The runtime is distributed: Windows handles UI/front-end checks, while Office Mac serves backend LLM/MLX operations.
- Canonical Agent IDs: G01, G02, G03, G04.
- Patch Grounding: A patch is fully grounded only when both a physical manifest exists and active-state linkage is verified.
- Streak Grounding: File on disk != live update; verify runtime wiring separately.

## 6. Gamma Scientific Workflow
- Log issues immediately to the JSONL issue log. Write markdown audits to the progress folder during major checkpoints.
- Distinguish `voltage-target-hit`, `gate-accepted`, and `accepted-streak-closed`.

## 7. Game Progression Constraints
- Active progression values must be loaded from receipt-backed Truth-plane state, not historical examples.
- Validation constraints: connected network required, no silent/flat/NaN/inf neurons, no pathological saturation, physiological voltage bounds, unlocks require biological validity.

## Reporting
Durable reports must begin and end exactly with:
`[model-llm-name][root-location][yyyymmdd-hhmm]`
Use verified values or `unknown_model_do_not_guess` / `unknown_root_do_not_guess`.
Include changed files, commands, validation evidence, truth status, blockers, and next safe action.

## ⚡ Fast-path doctrine (intelligence-per-token · per-watt) — all agents

Fast executive models: act on this block; read the rest only when blocked.
1. **Verify-before-call.** Never name a skill/API/flag/path/branch from memory — grep `__all__` / the contract first. Fabrication→fail→retry is the costliest loop.
2. **Contract-first.** Read the frozen contract (docstrings + `>TBI-not-ready`), not the 1000-line module.
3. **Skill = cache.** Run the skill; don't re-reason its checklist.
4. **Receipt, not justification.** Show `command` + `N passed`. Never report what you didn't run.
5. **Route by altitude.** Opus=architecture/contracts/truth-gates · Sonnet=implement vs contract+skill · Gemini=cross-file synthesis + batch edits.

JAX/jaxfne: `N_compile<=1` (recompilation is the silent watt+token sink) · stable shapes+dtypes · jit/vmap on (`runtime_report()`) · x64 at startup only · sparse>dense · `segment_sum`+`lax.scan` · pytree children=dynamic, aux=static. Depth → jaxfne skills + `~/.claude/CLAUDE.md`.
