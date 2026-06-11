# Masking Contract

This page defines the MVP masking conventions for NOPIMS MAE pretraining.

## Geometry

All spatial arrays use grid order `[x, y, z]`.

Production pretraining uses:

```text
local crop: [128, 128, 128]
context crop: [512, 512, 512]
context downsample: 4
context after downsample: [128, 128, 128]
patch size: [8, 8, 8]
token grid: [16, 16, 16]
```

The context crop is centered on the local crop, read with padding when it crosses
survey boundaries, then downsampled to the local crop shape. Downsampling uses
only valid in-survey voxels; padded voxels do not contribute to pooled values.
The returned `context_valid_mask` is `True` where a downsampled context cell had
at least one valid source voxel.

## Spatial Masks

`spatial_mask` is a boolean token mask with shape `[TX, TY, TZ]`, where:

```text
True = masked token to reconstruct
False = visible token
```

`visible_spatial_mask` has the same shape and is the inverse convention:

```text
True = visible token for encoder
False = masked token
```

The required relation is:

```text
visible_spatial_mask == ~spatial_mask
```

For the production geometry, `[128, 128, 128]` local crops and `[8, 8, 8]`
patches produce a `[16, 16, 16]` token grid.

## Attribute Masks

Attribute masks are 1D boolean arrays with one entry per MVP registry attribute.
They use stable registry order.

`attribute_input_mask` marks the selected input attributes:

```text
True = input-visible attribute
False = not provided to the encoder
```

`attribute_target_mask` marks valid reconstruction targets:

```text
True = valid reconstruction target
False = target is unavailable for this sample
```

`dropped_attribute_mask` marks valid target attributes withheld from the input:

```text
True = valid target attribute not visible as input
False = visible input attribute or invalid target
```

The required relation is:

```text
dropped_attribute_mask == attribute_target_mask & ~attribute_input_mask
```

The selected channel IDs must match the input mask:

```text
attribute_ids == np.flatnonzero(attribute_input_mask)
x.shape[0] == attribute_ids.shape[0]
```

## Attribute Dropout

The MVP starts from available manifest attributes and samples input-visible
attributes for each sample. `amplitude_norm` is required and always included.
Group dropout may remove a whole non-required attribute group when the minimum
input count can still be satisfied. Attribute dropout may remove individual
non-required attributes. The sampler then refills to `min_input_attributes` if
needed and trims to `max_input_attributes` if too many attributes remain.

The target mask is independent of dropout: it records whether a target attribute
exists in the manifest. Missing target attributes are not reconstruction targets.

## Scope

F3 is not used in pretraining. It is reserved for few-label fine-tuning and
held-out evaluation.

External structural model outputs, such as fault, channel, salt, or horizon
probability volumes, are not MVP pretraining inputs or targets.

The masked inpainting baseline is not part of the MVP.
