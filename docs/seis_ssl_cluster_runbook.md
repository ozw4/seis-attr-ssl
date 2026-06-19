# Seis SSL Cluster Runbook

Install the MVP extras:

```bash
python -m pip install -e ".[dev,cluster,visualization]"
```

## Manifest Build

```bash
python proc/seis_ssl_cluster/build_nopims_manifests.py \
  --config proc/configs/seis_ssl_cluster/build_nopims_manifests.yaml
```

This reads the explicit NOPIMS amplitude path-list configured at
`manifest.input_path_list` and writes an amplitude-only manifest under
`/workspace/artifacts/seis_ssl_cluster/registry/`.

## Stats Preparation

```bash
python proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py \
  --config proc/configs/seis_ssl_cluster/prepare_nopims_normalization_stats.yaml
```

This writes one robust normalization sidecar per manifest entry.

## QC Filtering

```bash
python proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py \
  --config proc/configs/seis_ssl_cluster/filter_manifest_by_normalization_qc.yaml
```

This writes the normalization QC report, excluded survey list, clean manifest,
and clean path-list.

## Two-Step Smoke Run

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cpu \
  --max-steps 2 \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/smoke_amp_mae
```

## 1000-Step Pilot

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cuda \
  --max-steps 1000 \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/pilot_amp_mae_1000
```

## Resume

```bash
python proc/seis_ssl_cluster/train_amp_mae.py \
  --config proc/configs/seis_ssl_cluster/train_amp_mae.yaml \
  --device cuda \
  --max-steps 1000 \
  --output-root /workspace/artifacts/seis_ssl_cluster/runs/pilot_amp_mae_1000 \
  --resume /workspace/artifacts/seis_ssl_cluster/runs/pilot_amp_mae_1000/mae_latest.pt
```

## Embedding Extraction

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config proc/configs/seis_ssl_cluster/extract_embeddings.yaml \
  --device cuda
```

Use `--skip-existing` to keep already-complete survey embeddings whose metadata
matches the requested run.

## Clustering

```bash
python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config proc/configs/seis_ssl_cluster/cluster_embeddings.yaml
```

The MVP uses MiniBatchKMeans on extracted amplitude MAE encoder embeddings.

## Visualization

```bash
python proc/seis_ssl_cluster/visualize_clusters.py \
  --config proc/configs/seis_ssl_cluster/visualize_clusters.yaml
```

This reconstructs voxel labels when configured and writes XY/XZ PNGs for token
and voxel cluster maps.
