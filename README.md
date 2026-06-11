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
- [Strict MAE pretraining](docs/mae_pretraining.md)
- [Masking contract](docs/masking.md)
- See [docs/data_pipeline.md](docs/data_pipeline.md) for the NOPIMS manifest and pretraining sample contract.

## Stage 1 MAE Pretraining

```bash
python proc/build_nopims_manifests.py --config proc/configs/build_nopims_manifests.yaml
python proc/train_mae.py --dry-run
python proc/train_mae.py --config proc/configs/mvp_mae.yaml --device cuda
```

Stage 1 uses external NOPIMS data only; F3 is reserved for fine-tuning and
evaluation. See [docs/mae_pretraining.md](docs/mae_pretraining.md) for the
batch contract, model shape contract, checkpoint contents, and smoke-test
command.
