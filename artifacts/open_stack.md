# open stack

Target: empty.

| ID | Severity | Symptom | Evidence | Fix item |
|---|---|---|---|---|
| P-1 | major | Root clean-up broke 2 passing tests | `PYTHONPATH=src python -m pytest tests/test_shared_root_checker.py -q`: 2 failed, `can't open file .../scripts/check_shared_root.py` | T-1 |
