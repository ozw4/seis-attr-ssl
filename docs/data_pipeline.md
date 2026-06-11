# NOPIMS Data Pipeline

This page records the data contract for NOPIMS source-seismic pretraining.
F3 is not used for pretraining; it is reserved for few-label fine-tuning and
held-out evaluation.

## Directory Structure

NOPIMS pretraining input is a survey directory tree containing dip-steered
median filtered source seismic `.npy` volumes and survey-wise robust
normalization stats:

```text
/home/dcuser/data/NOPIMS/
  survey_001/
    seismic/
      dip_steered_median_filtered.npy
    normalization_stats.json
  survey_002/
    seismic/
      dip_steered_median_filtered.npy
    normalization_stats.json
```

The default manifest scanner writes base-seismic manifest entries. The
pretraining dataset then generates MVP attributes on the fly from local and
context crops. Precomputed MVP 10-attribute volumes are not required for the
MVP path.

## Volume Contract

Each source seismic volume must be a numeric float32 3D `.npy` array readable
with:

```python
np.load(path, mmap_mode="r")
```

The grid order is `[x, y, z]`, so a NumPy volume has shape `[X, Y, Z]`. The
manifest records this as `shape_xyz` and `grid_order: ["x", "y", "z"]`.

Production pretraining defaults are:

```yaml
local_crop_size: [128, 128, 128]
context_crop_size: [512, 512, 512]
context_downsample: 4
local_attribute_halo: [16, 16, 64]
context_attribute_halo: [8, 8, 16]
require_full_halo_inside_volume: true
```

The local halo is in source-grid `[x, y, z]` coordinates. The context halo is
defined on the downsampled context grid, so its source-space margin is
`context_attribute_halo * context_downsample`. With the production defaults,
context attributes are generated on a source compute crop of
`[576, 576, 640]`, downsampled to `[144, 144, 160]`, and then center-trimmed
to the `[128, 128, 128]` context payload. Tests may use smaller crops and
volumes, and small synthetic volumes may fall back to ordinary crop sampling
when the full halo margin cannot fit. NOPIMS production pretraining assumes the
full halo fits inside the sampled volume.

## MVP Attributes

The MVP target attribute set is generated on the fly from survey-wise robust
normalized source seismic crops in stable channel order:

```yaml
attribute_names:
  - amplitude_norm
  - phase_sin
  - phase_cos
  - instantaneous_frequency
  - spectral_low_ratio
  - spectral_mid_ratio
  - spectral_high_ratio
  - coherence
  - glcm_contrast
  - glcm_homogeneity
```

External structural model outputs are not part of the MVP pretraining input.
The masked inpainting baseline is not part of the MVP.

## Manifest Creation

Build a NOPIMS manifest JSON with:

```bash
python proc/build_nopims_manifests.py --config proc/configs/build_nopims_manifests.yaml
```

The builder scans the configured NOPIMS root, writes a JSON list of survey
manifests, and stores source seismic metadata. In code, use
`build_nopims_base_seismic_manifests(...)` to scan and serialize, then
`read_manifest_json(...)` to reload the manifest list for datasets.

## Pretraining Sample Contract

`NopimsAttributePretrainDataset` returns a dictionary with these keys:

```text
x
target
attribute_ids
spatial_mask
visible_spatial_mask
attribute_input_mask
attribute_target_mask
dropped_attribute_mask
target_attribute_ids
valid_attributes
target_valid
coords
context
context_valid_mask
local_valid_mask
```

For one sample:

```text
x: selected on-the-fly input attributes, [C, X, Y, Z], float32
target: all on-the-fly MVP target attributes, [A, X, Y, Z], float32
attribute_ids: selected stable attribute IDs, int64
spatial_mask: MAE token mask, [TX, TY, TZ], bool, True means masked
visible_spatial_mask: visible MAE token mask, [TX, TY, TZ], bool
attribute_input_mask: selected input attributes over all A attributes, bool
attribute_target_mask: valid reconstruction targets over all A attributes, bool
dropped_attribute_mask: valid targets withheld from input, bool
target_attribute_ids: all stable MVP attribute IDs, int64
valid_attributes: validity flags for x channels, bool
target_valid: validity flags for target channels, bool
coords: survey ID, local payload start/size, local halo, compute start/size, and crop settings
context: selected on-the-fly context attributes after downsampling, [C, X, Y, Z], float32, or None
context_valid_mask: payload-only downsampled context validity mask, [X, Y, Z], bool, or None
local_valid_mask: local crop validity mask, [X, Y, Z], bool
```

The target attribute count `A` is the MVP registry size. Missing target
attributes are represented by `target_valid == False` for non-MVP manifests.
For source-seismic MVP manifests, all registry attributes are generated and
valid wherever the local crop is valid.

## Synthetic Smoke Test

The synthetic smoke test builds small `.npy` volumes under a temporary
NOPIMS-like directory, creates and reloads a manifest JSON, instantiates
`NopimsAttributePretrainDataset`, and fetches samples.

Run it with:

```bash
PYTHONPATH=src pytest -q tests/test_data_pipeline_smoke.py
```
