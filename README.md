# SeisAttrSSL

## Development

```bash
python -m pip install -e .[dev]
pytest
python -m compileall src proc
python proc/train_mae.py --dry-run
```

## Documentation

- [MVP specification](docs/mvp_spec.md)
