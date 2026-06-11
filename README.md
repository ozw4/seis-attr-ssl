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
- See [docs/data_pipeline.md](docs/data_pipeline.md) for the NOPIMS manifest and pretraining sample contract.
