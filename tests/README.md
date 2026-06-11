# Test Commands

Required review validation:

```bash
python -m compileall -q src proc tests
python -m ruff check .
pytest -q --ignore=tests/test_proc_dry_run.py
```

`tests/test_proc_dry_run.py` may still be run locally, but it is not part of
the required review command set.
