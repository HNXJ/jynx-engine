# todo stack

Top = next. Gate: `PYTHONPATH=src python -m pytest tests -q`.

- T-1 Point `tests/test_shared_root_checker.py` at `src/scripts/check_shared_root.py` (moved from `scripts/` in 7909d3a). Closes when: `PYTHONPATH=src python -m pytest tests/test_shared_root_checker.py -q` reports 0 failed. Blocked by: none.
- T-2 Fix `ModuleNotFoundError: src.gamma_runtime.types` raised from `src/sde_engine/solver.py` during collection of `tests/test_hub_api_status.py` and `tests/test_omission_stream.py`. Closes when: `PYTHONPATH=src python -m pytest tests/test_hub_api_status.py tests/test_omission_stream.py -q` collects with 0 errors. Blocked by: none.
- T-3 Fix `InferenceRequest.__init__() missing 'model_id'` in `src/apps/council_app.py:133`. Closes when: `PYTHONPATH=src python -m pytest tests/test_model_key_resolution.py tests/test_scientific_work_unit_prompt.py -q` reports 0 failed. Blocked by: none.
- T-4 Fix `object NoneType can't be used in 'await' expression` in the proposal emit path. Closes when: `PYTHONPATH=src python -m pytest tests/test_v1_gamma_sde_validation.py -q` reports 0 failed. Blocked by: none.
- T-5 Add a pytest config so `python -m pytest tests` runs without `PYTHONPATH=src`. Closes when: `python -m pytest tests -q` prints the same summary line as the gate. Blocked by: T-2.
- T-6 Remove or restore the `scripts/train_lora_mlx.py` reference in `src/gamma_runtime/consolidation.py:67,71` (file not in tree). Closes when: `git grep -n train_lora_mlx -- src` prints nothing or the path exists. Blocked by: none.
- T-7 Triage the 33 open issues (#81 to #100 and older); close those the archive move made moot. Closes when: `gh issue list -R HNXJ/jynx-engine --state open --json number -q length` prints 0. Blocked by: none.
