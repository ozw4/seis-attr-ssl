# Strict MAE Pretraining

This page documents the Stage 1 strict 3D attribute-set MAE pretraining path.

## Scope

Stage 1 uses external NOPIMS data only. F3 is not used for pretraining; it is
reserved for few-label fine-tuning and held-out evaluation.

The MVP pretraining input set is controlled by the explicit path list used to
build the NOPIMS manifest. Source seismic `.npy` memmaps listed there are the
only volumes used for Stage 1. The target set is the seismic attribute registry
in `proc/configs/mvp_mae.yaml`, generated on the fly during dataset sampling.
Phase targets use reflect-padded z-axis Hilbert phase, instantaneous frequency
uses a gated, smoothed, clipped z-phase gradient, and spectral ratio targets are
local z-window energy ratios that may vary along z. The GLCM targets remain
proxy texture channels.
Precomputed 10-attribute `.npy` volumes are not required. External structural
prediction attributes, such as fault, channel, salt, or horizon probability
volumes, are not pretraining inputs or targets. The masked inpainting baseline
is not part of the MVP.

## Batch Contract

After collation, the model and loss consume these core tensors:

```text
x: [B, C, X, Y, Z]
attribute_ids: [B, C]
attribute_valid_mask: [B, C]
target: [B, A, X, Y, Z]
spatial_mask: [B, TX, TY, TZ]
visible_spatial_mask: [B, TX, TY, TZ]
```

`C` is the padded count of input-visible attributes in the batch. `A` is the MVP
attribute registry size. Spatial arrays use `[x, y, z]` order. `spatial_mask` is
`True` for masked tokens that should be reconstructed, and
`visible_spatial_mask == ~spatial_mask`.

The training loss also uses `target_valid: [B, A]` and
`dropped_attribute_mask: [B, A]` to ignore missing target attributes and weight
valid attributes withheld from the encoder input. When context is enabled, the
batch may include `context: [B, C, X, Y, Z]` and
`context_valid_mask: [B, X, Y, Z]`.

For the default config, local crops are `[128, 128, 128]`, context crops are
`[256, 256, 512]`, context downsampling is `[2, 2, 4]`, patch size is
`[8, 8, 8]`, and the token grid is `[16, 16, 16]`.
On-the-fly targets are generated from halo-expanded compute crops and then
center-trimmed back to the payload shape before attribute subset sampling and
MAE spatial masking. The default local attribute halo is `[16, 16, 64]`; the
default context halo is `[8, 8, 16]` on the low-resolution context grid.
For NOPIMS volumes with maximum shape around `[300, 300, 1501]`, the
recommended context source requirement is `[288, 288, 640]`:

```text
[256, 256, 512] + 2 * [8, 8, 16] * [2, 2, 4]
```

Context can be disabled for local-only experiments with:

```yaml
data:
  use_context: false
```

## Dataset Sampling

`NopimsAttributePretrainDataset` samples each item deterministically from the
configured seed, current epoch, and dataset index. Re-reading the same
`(seed, epoch, index)` yields the same crop, attribute subset, and MAE spatial
mask, while advancing the epoch changes the random draw for that index. The MAE
training loop sets the dataset epoch before each training epoch.

`train.samples_per_epoch` controls how many random crop samples are drawn per
epoch. When it is omitted, the dataset length defaults to the number of survey
manifests, but each item is still an epoch-specific random crop sample rather
than a fixed survey-only record.

For NOPIMS-scale Stage 1 pretraining, `proc/configs/mvp_mae.yaml` defaults to
`train.samples_per_epoch: 10000`, `train.num_workers: 4`, and
`train.shuffle: true`. Small synthetic smoke-test configs can override these
values to keep local runs lightweight.

## Model

`StrictAttributeSetMAE3D` implements the pretraining model:

- `AttributePatchTokenizer3D` patchifies each selected attribute channel, adds
  stable attribute and group embeddings, and mean-fuses valid attribute channels
  into one token per spatial patch.
- The encoder receives visible spatial tokens only, selected by
  `visible_spatial_mask`.
- Optional context volumes are tokenized with the same attribute IDs, pooled by
  `ContextTokenPooler`, and appended as encoder context tokens.
- The decoder maps encoded visible tokens to decoder width, restores the full
  spatial token grid with learned mask tokens, optionally appends decoded context
  tokens, and predicts all target attributes for every spatial patch.

The primary model output is:

```text
pred_patches: [B, TX * TY * TZ, A, PX * PY * PZ]
```

where `[PX, PY, PZ]` is `model.patch_size`. The loss patchifies `target` and
computes reconstruction terms only on masked spatial patches and valid target
attributes.

## Commands

Build the manifest from the explicit path list:

```bash
python proc/build_nopims_manifests.py \
  --config proc/configs/build_nopims_manifests.yaml
```

Compute missing sidecar normalization stats for manifest entries:

```bash
python proc/prepare_nopims_normalization_stats.py \
  --config proc/configs/mvp_prepare_nopims_stats.yaml
```

## Phase 7.5 Stable Pilot

Phase 7.5 uses `proc/configs/mvp_mae_phase75_stable.yaml` for A100-oriented
clean-manifest pilot runs. The config pins the clean manifest path, output
directory, gradient clipping with `train.grad_clip_norm: 1.0`, step checkpoints
with `train.checkpoint_every_steps: 1000`, and non-finite diagnostic output
under `train.diagnostics_dir: diagnostics`. It intentionally omits
`train.max_steps`; set pilot length from the CLI.

Generate normalization stats:

```bash
python proc/prepare_nopims_normalization_stats.py \
  --config proc/configs/mvp_prepare_nopims_stats.yaml
```

Run stats QC and generate the clean manifest/split:

```bash
python proc/filter_nopims_manifest_by_stats_qc.py \
  --manifest registry/manifests/nopims/pretrain_v1/nopims_base_seismic_manifests.json \
  --input-path-list registry/splits/nopims/pretrain_v1/train_npy_paths.txt \
  --nopims-root /home/dcuser/data/NOPIMS \
  --output-qc-json registry/qc/nopims/pretrain_v1/normalization_stats_qc.json \
  --output-excluded-surveys registry/qc/nopims/pretrain_v1/excluded_surveys.txt \
  --output-manifest registry/manifests/nopims/pretrain_v1_clean/nopims_base_seismic_manifests.json \
  --output-path-list registry/splits/nopims/pretrain_v1_clean/train_npy_paths.txt \
  --iqr-min-threshold 1.0e-6 \
  --norm-abs-max-threshold 1.0e4
```

Confirm the stable config resolves to the clean manifest without touching data:

```bash
python proc/train_mae.py \
  --config proc/configs/mvp_mae_phase75_stable.yaml \
  --dry-run
```

Run a 1000-step pilot:

```bash
python proc/train_mae.py \
  --config proc/configs/mvp_mae_phase75_stable.yaml \
  --max-steps 1000
```

Resume from the latest checkpoint and continue to 10000 global steps:

```bash
python proc/train_mae.py \
  --config proc/configs/mvp_mae_phase75_stable.yaml \
  --resume runs/mae_nopims_pretrain_v1_clean_phase75/mae_latest.pt \
  --max-steps 10000
```

If a non-finite loss or gradient is detected, inspect:

```text
runs/mae_nopims_pretrain_v1_clean_phase75/diagnostics/nonfinite_mae_step_*.json
```

Record these fields when triaging the diagnostic JSON:

- `survey_id`
- `local_start_xyz`
- `local_compute_start_xyz`
- `context_compute_start_xyz`
- `attribute_ids`
- `losses`
- `tensors.x.all_finite`
- `tensors.target.all_finite`
- `tensors.context.all_finite`
- `tensors.pred_patches.all_finite`

Pilot checklist:

```text
[ ] normalization_stats_qc.json の excluded_surveys を確認した
[ ] clean manifest の survey count が 0 ではない
[ ] train dry-run が clean manifest path を指している
[ ] 1000 step pilot で NaN/Inf loss が出ない
[ ] mae_latest.pt が保存される
[ ] --resume mae_latest.pt で global_step が継続する
[ ] 10000 step pilot で loss が finite のまま推移する
[ ] diagnostics/ に nonfinite JSON が出ていない、または出た場合に原因 survey を特定できる
```

Smoke-test MAE pretraining:

```bash
python proc/train_mae.py \
  --config proc/configs/mvp_mae.yaml \
  --device cuda \
  --max-steps 2 \
  --output-root runs/smoke_mae
```

Run full MAE pretraining:

```bash
python proc/train_mae.py \
  --config proc/configs/mvp_mae.yaml \
  --device cuda \
  --output-root runs/mae_nopims
```

The dry-run test module is a local developer check. Required review validation
uses:

```bash
python -m compileall -q src proc tests
python -m ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
  pytest -q --ignore=tests/test_proc_dry_run.py
```

Run the implemented synthetic one-step smoke test:

```bash
pytest -q tests/test_proc_train_mae_integration.py::test_train_mae_proc_one_step_cpu_run_writes_checkpoint
```

## Outputs

`proc/train_mae.py` writes checkpoints under `paths.output_root`, which defaults
to `runs`. Checkpoint names are:

```text
mae_epoch_0001.pt
mae_epoch_0002.pt
...
```

Each checkpoint contains `model_state_dict`, `optimizer_state_dict`, `epoch`,
the resolved training `config`, `package_version`, and training `metrics`.
