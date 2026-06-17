# MVP Attribute Generation

MVP attributes are generated on the fly during dataset sampling from
survey-wise robust normalized source seismic crops. The source seismic is the
`.npy` memmap recorded in the path-list-derived base-seismic manifest.

The default local payload crop is `[128, 128, 128]`. Attribute generation reads
a larger compute crop and returns the unchanged payload crop shape:

```text
local base seismic payload: [128, 128, 128]
local attribute halo: [16, 16, 64]
local compute crop: [160, 160, 256]
```

Local halo-aware generation uses this order:

```text
base seismic compute crop
  -> survey-wise robust normalization
  -> on-the-fly MVP attribute generation
  -> center trim to [128, 128, 128]
  -> attribute subset sampling
  -> spatial mask for MAE
```

The recommended NOPIMS context payload crop is `[256, 256, 512]`, downsampled
by `[2, 2, 4]` to `[128, 128, 128]`. `context_downsample` may be either an
integer or an `[x, y, z]` list. Context attribute generation uses a halo of
`[8, 8, 16]` on the downsampled context grid; in source-space coordinates that
halo is multiplied by `context_downsample`:

```text
context source payload: [256, 256, 512]
context downsample: [2, 2, 4]
context low-res payload: [128, 128, 128]
context low-res halo: [8, 8, 16]
context source halo: [16, 16, 64]
context source compute crop: [288, 288, 640]
context low-res compute crop: [144, 144, 160]
```

Halo-aware on-the-fly generation uses this order:

```text
base seismic compute crop + halo
  -> survey-wise normalization
  -> downsample context crops when building context attributes
  -> generate attributes on compute crop
  -> center trim to payload crop
  -> attribute subset dropout / spatial mask
```

Spatial masking is applied only after attribute generation and center trimming.
Attributes must not be generated from spatially masked base seismic.

With `data.require_full_halo_inside_volume: true`, training samples require full
local and context halo coverage inside the source volume, and undersized volumes
are rejected. Set it to `false` only for small synthetic tests or experiments
that intentionally permit padded halo reads.

Set `data.use_context: false` to skip context crop reading and context attribute
generation for experiments that only use the local payload.

Generated attributes follow the stable `seis_attr_ssl` registry order:

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

## Attribute Meanings

`amplitude_norm` is the survey-wise robust normalized source seismic amplitude
used as the generator input.

`phase_sin` and `phase_cos` are the sine and cosine of instantaneous phase from
a Hilbert transform along the z axis. The transform uses reflect padding along z
and an optional symmetric taper before trimming back to the requested crop, which
reduces edge artifacts near crop boundaries.

`instantaneous_frequency` is derived from the z-axis gradient of unwrapped phase,
scaled to cycles per sample. It is envelope-gated, mean-smoothed along z, and
robust-clipped so low-amplitude phase instability does not dominate the channel.
The generated values are finite and non-negative after sanitization.

`spectral_low_ratio`, `spectral_mid_ratio`, and `spectral_high_ratio` are local
z-window spectral-energy ratios. They are not trace-global ratios: after
frequency-band filtering, band energies are measured in a moving z window, so
the ratios should vary along z when local frequency content changes. Where
nonzero energy is present, the three ratios are expected to approximately sum to
one.

`coherence` is a deterministic finite-difference similarity proxy.
`glcm_contrast` and `glcm_homogeneity` are quantized finite-difference texture
proxies, not full gray-level co-occurrence matrix estimates.

## Visualization QC

The on-the-fly comparison config in
`proc/configs/visualize_attribute_on_the_fly_compare.yaml` includes the
recommended `attribute_generation` defaults for checking reflect-padded phase,
stabilized instantaneous frequency, and local z-window spectral ratios. Use XZ
views to confirm spectral ratios change along z where the source trace changes
frequency content.

When using `use_known_ranges: true`, do not judge `glcm_homogeneity` only from a
fixed `[0, 1]` color range. The proxy can saturate near 1, so percentile-clipped
or focused QC views may be needed to see useful contrast.

## Known Limitations

- These attributes are deterministic MVP approximations.
- They are not a substitute for full commercial seismic attribute workflows.
- `glcm_contrast` and `glcm_homogeneity` are still proxy texture attributes and
  may need future replacement or rescaling.

Precomputed MVP 10-attribute volumes are not required for pretraining. External
structural prediction outputs are not used in the MVP, and the masked
inpainting baseline is not part of the MVP.
