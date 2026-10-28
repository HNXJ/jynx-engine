# Replay & reproducibility contract

A replay demonstrates that two executions under round **R_k** produce equivalent adjudication-relevant outputs.

## Environment lock

`RoundManifest.environment` carries:

- `environment_id` — logical environment name (for example Cloud Agent build id or conda lock name)
- `lock_digest` — sha256 of the canonical environment lock file(s)
- optional `container_image_digest` — OCI image digest when containers are used

Replay MUST use the same digests as pinned in the manifest; newer “latest” images are invalid for R_k.

## Input digests

All evaluation inputs appear in `evaluation_inputs[]` with `input_id` and `content_digest`. Replay bundles MUST list the same digests in run metadata before judges execute.

## Numerical tolerances

Simulation outputs compared during replay declare tolerances in rubric metadata (jynx-judge):

| Quantity kind | Default tolerance | Notes |
|---------------|-----------------|-------|
| Floating traces | max abs error ≤ 1e-6 | Unless rubric specifies otherwise |
| Integer counts | exact match | Neuron counts, event ordinals |
| Categorical labels | exact match | |

Judges emit PASS/FAIL on tolerance breaches without mutating evidence.

## Stochastic acceptance

When ENGINE uses stochastic seeds:

1. Seed MUST be recorded in evidence provenance (`simulation_run` parent receipt).
2. Replay runs use the same seed and JAX/NumPy global seed hooks.
3. Acceptance tests allow **two** modes:
   - **Deterministic**: bit-identical outputs required.
   - **Statistical**: rubric specifies minimum number of replicates and acceptable distribution distance (for example KS p-value threshold); judge issues typed outcome with evidence refs.

## Evidence & judgments

Replay compares:

- Ordered event ledger slice (WORLD projection hash)
- Judgment outcomes (`outcome.kind`, rubric version, evidence refs)
- Scoring vector **S** within rubric-defined bounds

Replay MUST NOT rewrite or delete prior evidence; contradictions append new receipts and judgments.

## Verification checklist

- [ ] Round manifest immutable flag true
- [ ] All pinned component SHAs and content digests match checkout
- [ ] Environment lock digest verified
- [ ] Evaluation input digests verified
- [ ] Held-out inputs never loaded on player code paths
- [ ] Judge outcomes reproducible within tolerances

Automated schema tests in this repository cover manifest pinning fields; full ENGINE replay harness lives in future **jynx-engine** work.
