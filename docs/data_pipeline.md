# NOPIMS Data Pipeline

This page records the data contract for NOPIMS source-seismic pretraining.
F3 is not used for pretraining; it is reserved for few-label fine-tuning and
held-out evaluation.

## Path List Input

NOPIMS pretraining input is controlled by an explicit text file of source
seismic `.npy` paths. Only files listed in this path list are used for
pretraining.

```text
# One .npy path per line.
# Empty lines and full-line comments are ignored.
/home/dcuser/data/NOPIMS/survey_001/seismic/base.npy
survey_002/seismic/base.npy
```

Relative paths are resolved against `paths.nopims_root`. The manifest builder
does not select volumes by `dip`, `median`, or `filtered` filename hints in the
MVP path-list workflow. Each manifest entry records one listed source volume,
and `survey_id` is generated deterministically from the relative path, for
example
`survey_a/seismic/base.npy -> survey_a__seismic__base`.

Each source volume has a sidecar robust-normalization stats file next to the
volume:

```text
volume.npy -> volume.normalization_stats.json
```

The pretraining dataset generates MVP attributes on the fly from local and
context crops. Precomputed MVP 10-attribute volumes are not required.

## Volume Contract

Each source seismic volume must be a numeric float32 3D `.npy` array readable
with:

```python
np.load(path, mmap_mode="r")
```

The grid order is `[x, y, z]`, so a NumPy volume has shape `[X, Y, Z]`. The
manifest records this as `shape_xyz` and `grid_order: ["x", "y", "z"]`.

Production pretraining defaults are configurable and recommended for NOPIMS
volumes with maximum shape around `[300, 300, 1501]`:

```yaml
local_crop_size: [128, 128, 128]
local_attribute_halo: [16, 16, 64]
use_context: true
context_crop_size: [256, 256, 512]
context_downsample: [2, 2, 4]
context_attribute_halo: [8, 8, 16]
require_full_halo_inside_volume: true
```

`local_crop_size` is the local payload size returned to the model.
`local_attribute_halo` is the extra source seismic margin read before local
attribute generation. `context_crop_size` is the wider source-seismic context
payload. `context_downsample` is an integer or `[x, y, z]` list that shrinks the
context payload to the local payload size. `context_attribute_halo` is the halo
used for attribute generation on the downsampled context grid; in source space,
it corresponds to `context_attribute_halo * context_downsample`.
`require_full_halo_inside_volume: true` rejects samples whose halo-expanded
compute crop would cross the source volume boundary.

With the recommended defaults, the local source crop required for attribute
generation is:

```text
local_crop_size + 2 * local_attribute_halo
[128, 128, 128] + 2 * [16, 16, 64] = [160, 160, 256]
```

The context source crop required for attribute generation is:

```text
context_crop_size + 2 * context_attribute_halo * context_downsample
[256, 256, 512] + 2 * [8, 8, 16] * [2, 2, 4] = [288, 288, 640]
```

The explicit path-list manifest records source-seismic `.npy` volumes. During
sampling, local and optional context crops are read from those source volumes
with halo margins, MVP attributes are generated on the fly, and the halo regions
are center-trimmed before tensors are returned. To disable context for a smaller
experiment, override:

```yaml
data:
  use_context: false
```

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

Phase channels are generated from a reflect-padded z-axis Hilbert transform,
instantaneous frequency is a stabilized z-phase-gradient channel, and spectral
ratio channels are local z-window energy ratios rather than trace-global
summaries. The GLCM channels remain deterministic proxy texture attributes.

## Manifest Creation

Build a NOPIMS manifest JSON with:

```bash
python proc/build_nopims_manifests.py --config proc/configs/build_nopims_manifests.yaml
```

The default config reads `manifest.input_path_list`, writes the manifest JSON
under `manifest.output_dir`, and stores source seismic metadata for each listed
volume. Use `read_manifest_json(...)` to reload the manifest list for datasets.

After building the manifest, compute any missing sidecar stats for its entries:

```bash
python proc/prepare_nopims_normalization_stats.py \
  --config proc/configs/mvp_prepare_nopims_stats.yaml
```

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
coords: survey ID, local payload start/size, local/context halo metadata, compute crop metadata, and crop settings
context: selected on-the-fly context attributes after downsampling, [C, X, Y, Z], float32, or None
context_valid_mask: payload-only downsampled context validity mask, [X, Y, Z], bool, or None
local_valid_mask: local crop validity mask, [X, Y, Z], bool
```

`sample["coords"]` includes:

```text
local_start_xyz
local_size_xyz
local_attribute_halo_xyz
local_compute_start_xyz
local_compute_size_xyz
context_size_xyz
context_attribute_halo_xyz
context_compute_start_xyz
context_compute_size_xyz
context_lowres_compute_size_xyz
context_downsample
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
