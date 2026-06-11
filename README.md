# SeisAttrSSL

## Development

```bash
python -m pip install -e .[dev]
python -m compileall -q src proc tests
python -m ruff check .
pytest -q --ignore=tests/test_proc_dry_run.py
```

`tests/test_proc_dry_run.py` remains available for local checks, but it is not
part of the required review validation command set.

## Documentation

- [MVP specification](docs/mvp_spec.md)
- [Strict MAE pretraining](docs/mae_pretraining.md)
- [Data format](docs/data_format.md)
- [Attribute generation](docs/attribute_generation.md)
- [Masking contract](docs/masking.md)
- See [docs/data_pipeline.md](docs/data_pipeline.md) for the NOPIMS manifest and pretraining sample contract.

## Stage 1 MAE Pretraining

```bash
python proc/build_nopims_manifests.py --config proc/configs/build_nopims_manifests.yaml
python proc/train_mae.py --dry-run
python proc/train_mae.py --config proc/configs/mvp_mae.yaml --device cuda
```

Stage 1 uses external NOPIMS data only; F3 is reserved for fine-tuning and
evaluation. The default MVP path reads dip-steered median filtered `.npy`
memmap source seismic under `/home/dcuser/data/NOPIMS/` and generates MVP
attributes on the fly during sampling; precomputed attribute volumes are not
required. See [docs/mae_pretraining.md](docs/mae_pretraining.md) for the batch
contract, model shape contract, checkpoint contents, and smoke-test command.
