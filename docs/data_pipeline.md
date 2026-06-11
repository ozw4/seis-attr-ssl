# NOPIMS Data Pipeline

This page records the data contract for NOPIMS attribute-volume pretraining.
F3 is not used for pretraining; it is reserved for few-label fine-tuning and
held-out evaluation.

## Directory Structure

NOPIMS pretraining input is a survey directory tree containing generated MVP
attribute volumes as `.npy` files:

```text
/home/dcuser/data/NOPIMS/
  survey_001/
    attributes/
      amplitude_norm.npy
      phase_sin.npy
      phase_cos.npy
      instantaneous_frequency.npy
      spectral_low_ratio.npy
      spectral_mid_ratio.npy
      spectral_high_ratio.npy
      coherence.npy
      glcm_contrast.npy
      glcm_homogeneity.npy
  survey_002/
    attributes/
      amplitude_norm.npy
      ...
```

The manifest scanner accepts files whose stem is an MVP attribute name, such as
`attributes/amplitude_norm.npy`. It also accepts layouts where a parent directory
is the attribute name and the file has another stem, such as
`phase_cos/volume.npy`.

## Volume Contract

Each attribute volume must be a numeric 3D `.npy` array readable with:

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
```

Tests may use smaller crops and volumes.

## MVP Attributes

The MVP input is exactly the generated seismic attribute set below, in stable
channel order:

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

## Manifest Creation

Build a NOPIMS manifest JSON with:

```bash
python proc/build_nopims_manifests.py --config proc/configs/build_nopims_manifests.yaml
```

The builder scans the configured NOPIMS root, writes a JSON list of survey
manifests, and stores attribute records in registry order. In code, use
`build_nopims_manifests(...)` to scan and serialize, then `read_manifest_json(...)`
to reload the manifest list for datasets.

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
x: selected input attributes, [C, X, Y, Z], float32
target: all MVP target attributes, [A, X, Y, Z], float32
attribute_ids: selected stable attribute IDs, int64
spatial_mask: MAE token mask, [TX, TY, TZ], bool, True means masked
visible_spatial_mask: visible MAE token mask, [TX, TY, TZ], bool
attribute_input_mask: selected input attributes over all A attributes, bool
attribute_target_mask: valid reconstruction targets over all A attributes, bool
dropped_attribute_mask: valid targets withheld from input, bool
target_attribute_ids: all stable MVP attribute IDs, int64
valid_attributes: validity flags for x channels, bool
target_valid: validity flags for target channels, bool
coords: survey ID, local crop start and crop settings
context: selected context attributes after downsampling, [C, X, Y, Z], float32, or None
context_valid_mask: downsampled context validity mask, [X, Y, Z], bool, or None
local_valid_mask: local crop validity mask, [X, Y, Z], bool
```

The target attribute count `A` is the MVP registry size. Missing target
attributes are represented by `target_valid == False`; `amplitude_norm` is
required for pretraining samples.

## Synthetic Smoke Test

The synthetic smoke test builds small `.npy` volumes under a temporary
NOPIMS-like directory, creates and reloads a manifest JSON, instantiates
`NopimsAttributePretrainDataset`, and fetches samples.

Run it with:

```bash
PYTHONPATH=src pytest -q tests/test_data_pipeline_smoke.py
```
