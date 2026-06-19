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
- [Seis SSL Cluster MVP specification](docs/seis_ssl_cluster_mvp_spec.md)
- [Seis SSL Cluster runbook](docs/seis_ssl_cluster_runbook.md)
- [Seis SSL Cluster artifact layout](docs/seis_ssl_cluster_artifact_layout.md)
- [Strict MAE pretraining](docs/mae_pretraining.md)
- [Phase 7.5 stable MAE pilot](docs/mae_pretraining.md#phase-75-stable-pilot)
- [MAE debug visualization](docs/mae_debug_visualization.md)
- [Data format](docs/data_format.md)
- [Attribute generation](docs/attribute_generation.md)
- [Masking contract](docs/masking.md)
- See [docs/data_pipeline.md](docs/data_pipeline.md) for the NOPIMS manifest and pretraining sample contract.

## Stage 1 MAE Pretraining

```bash
python proc/build_nopims_manifests.py \
  --config proc/configs/build_nopims_manifests.yaml
python proc/prepare_nopims_normalization_stats.py \
  --config proc/configs/mvp_prepare_nopims_stats.yaml
python proc/train_mae.py \
  --config proc/configs/mvp_mae.yaml \
  --device cuda \
  --max-steps 2 \
  --output-root runs/smoke_mae
python proc/train_mae.py \
  --config proc/configs/mvp_mae.yaml \
  --device cuda \
  --output-root runs/mae_nopims
```

Stage 1 uses external NOPIMS data only; F3 is reserved for fine-tuning and
evaluation. The default MVP path builds its manifest from an explicit text file
of source-seismic `.npy` paths. Relative paths are resolved against
`paths.nopims_root`, only listed files are used, sidecar stats are stored as
`volume.normalization_stats.json`, and MVP attributes are generated on the fly
during sampling. Precomputed attribute volumes are not required. See
[docs/mae_pretraining.md](docs/mae_pretraining.md) for the batch contract, model
shape contract, checkpoint contents, and smoke-test command.

Default NOPIMS geometry uses `local_crop_size: [128, 128, 128]`,
`context_crop_size: [256, 256, 512]`, and `context_downsample: [2, 2, 4]`.
Context is configurable per experiment and can be disabled with
`data.use_context: false`.

## Seis SSL Cluster MVP

The in-repository `seis_ssl_cluster` package is an amplitude-only clustering
MVP intended for later extraction into a standalone `seis-ssl-cluster`
repository. Its exact path is:

```text
explicit NOPIMS amplitude path-list
-> amplitude manifest
-> normalization stats
-> normalization QC and clean manifest
-> amplitude-only MAE
-> embedding extraction
-> MiniBatchKMeans
-> cluster XY/XZ visualization
```

Use `/workspace/artifacts/seis_ssl_cluster/registry/` for canonical registry
inputs and `/workspace/artifacts/seis_ssl_cluster/runs/` for run outputs. The
MVP excludes fixed seismic attributes during pretraining, the attribute
registry, F3 supervised fine-tuning, dense adaptation, context branch, and DDP.
See the runbook for the full command sequence.
