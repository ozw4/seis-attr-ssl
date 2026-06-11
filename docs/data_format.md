# MVP Data Format

MVP pretraining data lives under `/home/dcuser/data/NOPIMS/` and consists of
approximately 100 NOPIMS-derived 3D surveys. F3 is not pretraining data; it is
reserved for fine-tuning and held-out evaluation.

Source volumes are dip-steered median filtered seismic stored as `.npy` memmaps
read with `np.load(path, mmap_mode="r")`. The grid order is `[x, y, z]`, so
NumPy volumes use shape `[X, Y, Z]`.

Default MVP geometry:

```yaml
input_format: npy_memmap
grid_order: [x, y, z]
source_seismic_kind: dip_steered_median_filtered
attribute_mode: on_the_fly
local_crop_size: [128, 128, 128]
context_crop_size: [512, 512, 512]
context_downsample_factor: 4
context_after_downsample: [128, 128, 128]
```

Only survey-wise robust normalization is applied before attribute generation.
Smooth time/depth trend correction, trace-wise AGC, patch-wise z-score, and
local whitening are outside the MVP.
