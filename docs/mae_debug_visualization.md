# MAE Debug Visualization

MAE debug visualization writes PNG grids during Stage 1 pretraining so a pilot
run can inspect whether masked reconstruction is learning sensible attribute
structure. Use `proc/configs/mvp_mae_debug_visualization.yaml` as the copyable
A100/NOPIMS example.

Each trigger writes one XY image and one XZ image for each selected sample. The
default slice indices are `null`, which renders the center z slice for XY and
the center y slice for XZ.

## Columns

- `input`: the attribute channel provided to the MAE encoder before spatial
  masking.
- `masked_input`: the same channel with masked spatial tokens hidden.
- `target`: the on-the-fly attribute target used by the reconstruction loss.
- `prediction`: the unpatchified MAE reconstruction for the target attribute.
- `abs_error`: absolute reconstruction error between `prediction` and `target`.

The example config renders these rows:

```yaml
attributes:
  - amplitude_norm
  - phase_sin
  - instantaneous_frequency
  - spectral_mid_ratio
  - coherence
  - glcm_contrast
```

## Mask Panels

When `show_spatial_mask_panel: true`, the renderer adds the MAE spatial mask as
a panel. Masked regions are the token locations reconstructed by the decoder;
visible regions are the token locations available to the encoder.

When `show_valid_mask_panel: true`, the renderer adds the displayed slice of
`local_valid_mask`. This mask comes from the zero-amplitude valid-mask workflow
and marks voxels eligible for reconstruction loss. With
`mask_invalid_values: true`, invalid voxels are also hidden in attribute panels
so loss-excluded regions are visually distinct from reconstruction errors.

## Output Location

Set `visualization.mae_debug.output_dir` to a path to control the PNG directory.
When it is `null`, files are written beside the checkpoint root:

```text
<paths.output_root>/../visualizations/mae_debug
```

For example, this output root:

```text
/workspace/artifacts/ssl/runs/<run_id>/checkpoints_debug_vis
```

writes PNGs to:

```text
/workspace/artifacts/ssl/runs/<run_id>/visualizations/mae_debug
```

File names include epoch, global step, sample index, and view suffix, such as
`mae_debug_epoch_0001_step_000100_sample_00_xy.png`.

## Frequency

Use step-based output for pilots:

```yaml
visualization:
  mae_debug:
    every_n_steps: 100
    every_n_epochs: null
    max_batches_per_trigger: 1
    max_samples_per_batch: 1
```

`every_n_steps: 100` is a practical starting point for a 1000-step debug run.
Increase the interval for longer training jobs to avoid excessive PNG output.

## Pilot Training Settings

The example config uses these A100/NOPIMS pilot settings:

```yaml
train:
  amp: false
  lr: 3.0e-5
  batch_size: 4
  num_workers: 8
```

## Example Command

```bash
python proc/train_mae.py \
  --config proc/configs/mvp_mae_debug_visualization.yaml \
  --device cuda \
  --max-steps 1000 \
  --output-root /workspace/artifacts/ssl/runs/<run_id>/checkpoints_debug_vis
```

Validate the config without training:

```bash
python proc/train_mae.py \
  --config proc/configs/mvp_mae_debug_visualization.yaml \
  --dry-run
```
