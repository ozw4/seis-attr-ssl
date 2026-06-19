# Seis SSL Cluster MVP Specification

## Scope

`seis_ssl_cluster` is an in-repository MVP package for amplitude-only seismic
SSL clustering. It is intentionally structured so a later task can copy it into
a standalone `seis-ssl-cluster` repository without importing `seis_attr_ssl`.

The exact MVP path is:

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

The MVP excludes:

```text
fixed seismic attributes during pretraining
attribute registry
F3 supervised fine-tuning
dense adaptation
context branch
DDP
```

## Canonical Inputs

The canonical human-maintained input is an explicit `.txt` path-list of NOPIMS
amplitude `.npy` files. Relative entries are resolved against
`paths.nopims_root`. The manifest build step records each survey's amplitude
path, XYZ shape, dtype, grid order, and planned normalization stats sidecar.

Canonical generated registry files are:

- `registry/splits/nopims/<split>/train_npy_paths.txt`
- `registry/manifests/nopims/<split>/nopims_amplitude_manifests.json`
- `registry/normalization_stats/nopims/<split>/*.normalization_stats.json`
- `registry/qc/nopims/<split>/normalization_stats_qc.json`
- `registry/qc/nopims/<split>/excluded_surveys.txt`

The clean manifest and clean path-list are canonical inputs to MAE training.
Embeddings, cluster labels, voxel reconstructions, summaries, and PNGs are
derived from a training checkpoint and can be regenerated.

## Reproducibility Contract

Every training run captures:

- resolved config: `resolved_config.json`
- input split snapshot: `inputs/<path-list-name>`
- manifest snapshot: `manifest.json`
- checkpoint/model reference: `mae_epoch_*.pt`, `mae_step_*.pt`, and
  `mae_latest.pt`
- git commit when available: `run_metadata.json`
- seed: in `resolved_config.json`
- package version: checkpoint payload and `run_metadata.json`
- output metadata: checkpoint metrics/training state and extraction/clustering
  metadata JSON files

Embedding extraction records the checkpoint path, checkpoint SHA-256, model
geometry, patch size, source amplitude path, source normalization stats path,
volume shape, token grid shape, window geometry, zero-mask settings, output
dtype, and token validity threshold.

Clustering records embedding inputs, embedding metadata hashes, compatibility
signature, sample counts, random seed, preprocessing settings, MiniBatchKMeans
settings, cluster counts, and per-survey label metadata.
