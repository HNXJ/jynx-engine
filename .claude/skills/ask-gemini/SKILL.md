---
name: ask-gemini
description: Delegate a programming task to Gemini CLI as the back-agent executor. Use when the user explicitly says "ask gemini", "use the back agent", "delegate", or when a task involves repo-scale mechanical edits, large-context cross-file reading (>50K tokens), or batch generation that would otherwise consume excessive main-context tokens.
allowed-tools: Read, Grep, Glob, Bash, Task
---

# Ask Gemini — Back-Agent Delegation Skill

Dispatch programming work to Gemini CLI via the `gemini-worker` subagent.

## When to Invoke

- User explicitly requests Gemini / back-agent / "delegate this".
- Task is mechanical: rename across N files, apply a known pattern to many sites, batch-generate boilerplate, regenerate fixtures.
- Task requires reading a very large slice of the repo (Gemini's 1M context wins here).
- Cost: this task is cheap-executor work (Haiku tier in the Gamma model economy).

## When NOT to Invoke

- Architecture decisions, science-of-record judgments, truth-plane commits → keep in Claude (Opus).
- Anything touching `truth/`, `receipts/`, `solver/`, `adapter/`, `runtime-state/` → keep in Claude with explicit auth.
- Anything credential-adjacent → never delegate.

## Protocol

1. Parse the user request into: `{mode, repo_dir, mission, allowed_read, allowed_write, validation_gates}`.
   - **mode**: `plan` (default), `audit`, or `write`
   - **repo_dir**: current directory or user-specified
   - **mission**: 1–3 sentence task description
   - **allowed_read**: list of files/patterns Gemini may read
   - **allowed_write**: list of files/patterns Gemini may edit (for write mode only)
   - **validation_gates**: tests, lints, or checks that must pass

2. Use the `Task` tool to dispatch to the `gemini-worker` subagent with that struct.

3. When the subagent returns, do a THETA pass:
   - Confirm files-touched matches `allowed_write`.
   - Confirm receipt path exists and exit code = 0.
   - If `mode=plan`, present the diff to the user for explicit approval before any write.

4. Final user-facing summary uses the standard A–I report format with the footer.

## Default Arguments Shape

When called as `/ask-gemini`, expect `$ARGUMENTS` to be free-form. Parse it. If repo_dir is omitted, assume the current working directory. If mode is omitted, default to `plan`. **Never default to `write`.**

## Example Usage

```
/ask-gemini  Rewrite all logger.info() calls in src/runtime/ to use the new
             structured logger from src/lib/log.py. plan mode. read-only.
```

What happens:

1. This skill parses the user input.
2. Dispatches to `gemini-worker` subagent via `Task` tool.
3. Subagent writes a prompt file and runs the wrapper: `~/.gamma/bin/gamma-gemini.sh plan <repo> <prompt>`.
4. Gemini reads the repo (1M context), proposes a unified diff, exits.
5. Receipt written to `~/.gamma/receipts/gemini/20260516T...json`.
6. Subagent returns the A–I report with the diff inline.
7. Main Claude shows you the diff, waits for your "apply it" before doing actual `Edit` calls.

For `write` mode, you'd say `…apply it. write mode authorized.` and the subagent re-invokes with `GAMMA_GEMINI_WRITE=1 ~/.gamma/bin/gamma-gemini.sh write …`.

## Safety & Credential Notes

- The wrapper is the only thing that ever sees env vars or auth state.
- The subagent's `allowed-tools` is `Bash`, but in practice it only ever calls one script — you can tighten further by setting an `allow` pattern in `.claude/settings.json`:
  ```json
  {
    "permissions": {
      "allow": ["Bash(~/.gamma/bin/gamma-gemini.sh:*)"]
    }
  }
  ```

- `--yolo` is gated behind both `mode=write` AND `GAMMA_GEMINI_WRITE=1`. Default behaviour is read-only, so a misrouted task can't mutate the repo.

- Receipts live outside the repo (`~/.gamma/receipts/`) so they don't pollute `git status` or leak into PRs. If you want them committed for audit, symlink a subset into a `gamma-arena` evidence folder — but keep `stderr` files local.

- **Truth-plane safety:** the subagent's prompt explicitly forbids Gemini from touching `truth/`, `receipts/`, `solver/`, `adapter/`, `runtime-state/`. This is prompt discipline, not a hard boundary.

## Optional Upgrade Path: MCP Server

When this matures and you want structured tool calls instead of free-form bash + parsing, wrap `gamma-gemini.sh` in a tiny MCP server exposing tools like `gemini.plan`, `gemini.audit`, `gemini.write`. Then both Claude Code and any other MCP-aware client get a typed interface, and you get per-tool permissions in `.claude/settings.json`. The `mcp-builder` skill covers the build pattern — hold off until the bash version has earned a real workflow first.
