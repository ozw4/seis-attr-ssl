# MVP Data Format

MVP pretraining data lives under `/home/dcuser/data/NOPIMS/` and consists of
approximately 100 NOPIMS-derived 3D surveys. F3 is not pretraining data; it is
reserved for fine-tuning and held-out evaluation.

Source volumes are listed explicitly in the NOPIMS path-list file and stored as
`.npy` memmaps read with `np.load(path, mmap_mode="r")`. Relative path-list
entries are resolved against `paths.nopims_root`. The grid order is
`[x, y, z]`, so NumPy volumes use shape `[X, Y, Z]`.

Default MVP data geometry:

```yaml
grid_order: [x, y, z]
volume_format: npy_memmap
attribute_mode: on_the_fly
manifest_source: explicit_path_list
manifest.input_path_list: /home/dcuser/data/NOPIMS/train_npy_paths.txt
manifest.normalization_stats_suffix: .normalization_stats.json
local_crop_size: [128, 128, 128]
local_attribute_halo: [16, 16, 64]
use_context: true
context_crop_size: [256, 256, 512]
context_downsample: [2, 2, 4]
context_attribute_halo: [8, 8, 16]
require_full_halo_inside_volume: true
```

The recommended NOPIMS context crop `[256, 256, 512]` is downsampled by
`[2, 2, 4]` to `[128, 128, 128]`. `context_downsample` may be a scalar integer
or an `[x, y, z]` list. For local-only experiments, set `use_context: false`.

Only survey-wise robust normalization is applied before attribute generation.
Smooth time/depth trend correction, trace-wise AGC, patch-wise z-score, and
local whitening are outside the MVP.
