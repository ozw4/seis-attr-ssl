# SeisAttrSSL

## Development

```bash
python -m pip install -e .[dev]
ruff check .
pytest
python -m compileall src proc tests
python proc/train_mae.py --dry-run
```

## Documentation

- [MVP specification](docs/mvp_spec.md)
