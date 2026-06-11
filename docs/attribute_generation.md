# MVP Attribute Generation

MVP attributes are generated on the fly during dataset sampling from
survey-wise robust normalized source seismic crops. The source seismic is the
dip-steered median filtered `.npy` memmap recorded in the base-seismic manifest.

The local crop is `[128, 128, 128]`. The context payload crop is
`[512, 512, 512]`, downsampled by 4 to `[128, 128, 128]`. Attribute generation
uses a local halo of `[16, 16, 64]` around the payload crop. Context attribute
generation uses a halo of `[8, 8, 16]` on the downsampled context grid; in
source-space coordinates that halo is multiplied by `context_downsample`.
With the production defaults, the context source compute crop is
`[576, 576, 640]`, the low-resolution compute crop is `[144, 144, 160]`, and
only the center `[128, 128, 128]` payload is returned.

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

Training samples require full halo coverage inside the volume whenever
possible. Small synthetic test volumes may fall back to ordinary payload crop
sampling when the full margin cannot fit; NOPIMS production pretraining assumes
the full local and context halo fit inside each sampled volume.

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

Precomputed MVP 10-attribute volumes are not required for pretraining. External
structural prediction outputs are not used in the MVP, and the masked
inpainting baseline is not part of the MVP.
