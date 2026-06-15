# SeisAttrSSL MVP Specification

This document records the fixed MVP specification for the `ozw4/seis-attr-ssl` repository.
Downstream implementation work should treat these decisions as the project baseline.

## 1. Objective

The MVP objective is:

```text
F3を事前学習に使わず、NOPIMSから収集した多数の3D seismic surveyで自己教師あり事前学習した3D属性表現が、F3への少数ラベルfine-tuningにおいてfrom scratchより有効かを検証する。
```

Project identifiers:

```text
Repository: ozw4/seis-attr-ssl
Display name: SeisAttrSSL
Python package: seis_attr_ssl
Repository/package name used in code: seis_attr_ssl
```

## 2. Fixed Decisions

F3 is not used for pretraining. It is reserved for the fine-tuning task and held-out evaluation target.

The main evaluation name is:

```text
few-label seismic facies fine-tuning on F3
```

The main comparison is:

```text
A. from scratch on F3 few labels
B. external strict MAE pretrain -> F3 few-label fine-tune
C. external strict MAE pretrain -> external dense adaptation -> F3 few-label fine-tune
```

The primary result is:

```text
C vs A
```

## 3. Data Assumptions

Pretraining data:

```text
Data source: NOPIMS collected seismic surveys
Scale: approximately 100 3D surveys
Data root: /home/dcuser/data/NOPIMS/
Manifest source: explicit .npy path-list file
Base seismic: listed source-seismic .npy volumes
File format: .npy memmap
Reader: np.load(path, mmap_mode="r")
Manifest kind: source base seismic manifest
```

Fine-tuning and evaluation data:

```text
F3 is not used for pretraining.
F3 is used only as the fine-tuning task and held-out evaluation target.
Main evaluation name: few-label seismic facies fine-tuning on F3.
```

Hardware assumption:

```text
GPU: NVIDIA A100
GPU memory shortage fallback: out of scope for MVP
Crop size fallback: out of scope for MVP
```

## 4. Grid and Tensor Conventions

Grid and tensor order:

```text
Grid order: [x, y, z]
Numpy volume shape: [X, Y, Z]
Torch sample x: [C, X, Y, Z]
Torch sample target: [A, X, Y, Z]
Batch x: [B, C, X, Y, Z]
Batch target: [B, A, X, Y, Z]
```

Crop sizes:

```yaml
local_crop_size: [128, 128, 128]
context_crop_size: [512, 512, 512]
context_downsample: 4
context_after_downsample: [128, 128, 128]
```

Halo-aware on-the-fly attribute defaults:

```yaml
data:
  local_attribute_halo: [16, 16, 64]
  context_attribute_halo: [8, 8, 16]
  require_full_halo_inside_volume: true
```

## 5. Attributes and Normalization

MVP attributes are generated on the fly from normalized source seismic crops.
Precomputed 10-attribute volumes are not required for Stage 1 pretraining.

Pre-attribute normalization:

```text
Pre-attribute normalization: survey-wise robust normalization only.
Do not use smooth time/depth trend correction.
Do not use trace-wise AGC.
Do not use patch-wise z-score.
Do not use local whitening.
```

Recommended pre-attribute normalization:

```yaml
pre_attribute_normalization:
  clipping_percentiles: [0.5, 99.5]
  center: median
  scale: iqr
  epsilon: 1.0e-6
  smooth_time_depth_trend_correction: false
  trace_wise_agc: false
  patch_wise_zscore: false
```

MVP attribute dictionary:

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

attribute_groups:
  amplitude_norm: waveform
  phase_sin: phase
  phase_cos: phase
  instantaneous_frequency: frequency
  spectral_low_ratio: spectral
  spectral_mid_ratio: spectral
  spectral_high_ratio: spectral
  coherence: discontinuity
  glcm_contrast: texture
  glcm_homogeneity: texture
```

## 6. MVP Pipeline

The main pipeline is:

```text
NOPIMS 3D seismic surveys
  -> explicit path-list of source-seismic .npy volumes
  -> sidecar robust normalization stats per listed volume
  -> survey-wise robust normalization
  -> on-the-fly MVP attribute generation from local/context crops
  -> strict 3D attribute-set MAE pretraining
  -> external-data dense adaptation
  -> F3 few-label fine-tuning
  -> F3 held-out evaluation
```

## 7. Training Stages

The MVP uses three training and comparison paths:

```text
A. from scratch on F3 few labels
B. external strict MAE pretrain -> F3 few-label fine-tune
C. external strict MAE pretrain -> external dense adaptation -> F3 few-label fine-tune
```

Path C is compared against path A as the primary result.

Stage 1 strict MAE pretraining defaults in `proc/configs/mvp_mae.yaml` are set
for initial NOPIMS-scale pretraining:

```yaml
train:
  batch_size: 1
  samples_per_epoch: 10000
  num_workers: 4
  shuffle: true
```

Synthetic smoke-test configs may override these training values to use shorter
epochs or single-process loading.

The default pretraining manifest is a source-seismic manifest such as
`/home/dcuser/data/NOPIMS/manifests/nopims_base_seismic_manifests.json`.
Manifest entries point to the `.npy` source volumes from the explicit path list
and their sidecar robust normalization stats; they do not need precomputed
attribute paths.

## 8. F3 Evaluation Protocol

F3 must not be included in pretraining data. F3 is used only for few-label seismic facies fine-tuning and held-out evaluation.

The evaluation protocol should report the main evaluation as:

```text
few-label seismic facies fine-tuning on F3
```

## 9. Repository Layout

Repository-level project identifiers are:

```text
Repository: ozw4/seis-attr-ssl
Display name: SeisAttrSSL
Python package: seis_attr_ssl
```

The MVP specification lives at:

```text
docs/mvp_spec.md
```

## 10. Out of Scope

The following are excluded from the MVP:

```text
- FaultSeg output
- channel probability output
- salt probability output
- horizon probability output
- other external structural model outputs
- physical-amplitude-dependent attributes
- AVO attributes
- inversion attributes
- masked inpainting baseline
```

The following implementation work is also outside this issue:

```text
- config loader implementation
- YAML config creation
- data loader implementation
- model implementation
```
