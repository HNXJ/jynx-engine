# state

Stamp: jynx-engine@main@4f7cddb · 2026-10-28

## Gate (Python 3, `pip install -r requirements-dev.txt`)

`PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors`: 7 failed, 55 passed, 2 errors.
- Errors: test_hub_api_status, test_omission_stream (T-2).
- Failed: test_shared_root_checker x2 (T-1), test_model_key_resolution x1 and test_scientific_work_unit_prompt x1 (T-3), test_v1_gamma_sde_validation x3 (T-4).
- Without `PYTHONPATH=src`: 19 collection errors (T-5). Without the continue flag the run stops at the 2 collection errors.

## Works

- `python src/scripts/check_shared_root.py <repo>` prints OK for all seven game repos.
- Protocol layer: `tests/test_protocol_schemas.py`, `tests/test_tournament_validator.py` pass.

## Next

T-1.
