# Strict MAE Pretraining

This page documents the Stage 1 strict 3D attribute-set MAE pretraining path.

## Scope

Stage 1 uses external NOPIMS data only. F3 is not used for pretraining; it is
reserved for few-label fine-tuning and held-out evaluation.

The MVP pretraining input and target set is the generated seismic attribute
registry in `proc/configs/mvp_mae.yaml`. External structural prediction
attributes, such as fault, channel, salt, or horizon probability volumes, are
not pretraining inputs or targets. The masked inpainting baseline is not part of
the MVP.

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

For the default config, local crops are `[128, 128, 128]`, patch size is
`[8, 8, 8]`, and the token grid is `[16, 16, 16]`.

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

Build NOPIMS manifests:

```bash
python proc/build_nopims_manifests.py --config proc/configs/build_nopims_manifests.yaml
```

Validate the MAE config without training:

```bash
python proc/train_mae.py --dry-run
```

Run Stage 1 pretraining:

```bash
python proc/train_mae.py --config proc/configs/mvp_mae.yaml --device cuda
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

