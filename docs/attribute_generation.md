# MVP Attribute Generation

MVP attributes are generated on the fly during dataset sampling from
survey-wise robust normalized source seismic crops. The source seismic is the
dip-steered median filtered `.npy` memmap recorded in the base-seismic manifest.

The local crop is `[128, 128, 128]`. The context crop is `[512, 512, 512]`,
downsampled by 4 to `[128, 128, 128]` before context attributes are generated.

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
