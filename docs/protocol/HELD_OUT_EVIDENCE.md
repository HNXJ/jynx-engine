# Held-out evidence

Held-out evaluation data must be **inaccessible to players** during a round—not merely labeled “held out” in UI copy.

## Mechanism (reference design)

1. **Commitment** — Tournament authors register held-out items in `tournament.yaml` under `held_out_evidence` with:
   - `evidence_id` (`ev:<sha256>`)
   - `content_digest` (sha256 hex)
   - `access_policy`: `judge_sealed_commit` or `adjudicator_only`
   - `commitment_ref`: opaque id (no filesystem paths)

2. **Sealed registry** — Parallel file `tournaments/<id>/sealed/held_out_registry.json` lists the same commitments with `sealed: true` and `player_accessible: false`. Payload bytes live outside player-facing storage (judge enclave, WORM bucket, or encrypted blob keyed by adjudicator service).

3. **Access path** — Only judge plugins with adjudicator credentials may fetch bytes, verifying digest before use. ARENA/ENGINE/LAB clients receive **metadata only** (id + digest + policy).

4. **Validation** — `validate_tournament()` rejects tournaments when:
   - `commitment_ref` is missing from the sealed registry
   - `evidence_id` or digest mismatches the registry
   - registry marks `player_accessible: true`

5. **Round manifest** — `evaluation_inputs[]` may set `held_out: true`; ENGINE must refuse to hydrate those inputs without a judgment-gated capability token (future runtime work; documented here as contract).

Falsification and replication moves still cite **public** evidence ids; held-out sets are for scoring and blind evaluation only.
