# state

Stamp: jynx-engine@main@b53dd78 · 2026-10-28

## Gate (Python 3, `pip install -r requirements-dev.txt`)

`PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors`: 5 failed, 57 passed, 2 errors.
- Errors: test_hub_api_status, test_omission_stream (T-2).
- Failed: test_model_key_resolution x1 and test_scientific_work_unit_prompt x1 (T-3), test_v1_gamma_sde_validation x3 (T-4).
- Without `PYTHONPATH=src`: 19 collection errors (T-5). Without the continue flag the run stops at the 2 collection errors.

## Works

- `python src/scripts/check_shared_root.py <repo>` prints OK for all seven game repos; `tests/test_shared_root_checker.py` 2 passed.
- Protocol layer: `tests/test_protocol_schemas.py`, `tests/test_tournament_validator.py` pass.

## Next

T-2.
