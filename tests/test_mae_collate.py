from __future__ import annotations

import numpy as np
import torch

from seis_attr_ssl.models.mae import StrictAttributeSetMAE3D
from seis_attr_ssl.training import build_mae_dataloader
from seis_attr_ssl.training.collate import mae_collate_fn, move_batch_to_device


def _sample(
	attribute_ids: tuple[int, ...],
	*,
	use_context: bool,
	coords: dict[str, object] | None = None,
) -> dict[str, object]:
	channel_count = len(attribute_ids)
	shape = (2, 2, 2)
	sample: dict[str, object] = {
		'x': np.full((channel_count, *shape), channel_count, dtype=np.float32),
		'target': np.ones((4, *shape), dtype=np.float32),
		'attribute_ids': np.asarray(attribute_ids, dtype=np.int64),
		'spatial_mask': np.asarray([[[True]]], dtype=bool),
		'visible_spatial_mask': np.asarray([[[False]]], dtype=bool),
		'attribute_input_mask': np.asarray([True, False, True, False], dtype=bool),
		'attribute_target_mask': np.ones(4, dtype=bool),
		'dropped_attribute_mask': np.asarray([False, True, False, True], dtype=bool),
		'valid_attributes': np.ones(channel_count, dtype=bool),
		'target_valid': np.asarray([True, True, False, True], dtype=bool),
		'coords': coords or {'survey_id': f'survey-{channel_count}'},
		'context': None,
		'context_valid_mask': None,
	}
	if use_context:
		sample['context'] = np.full(
			(channel_count, *shape),
			channel_count + 10,
			dtype=np.float32,
		)
		sample['context_valid_mask'] = np.ones(shape, dtype=bool)
	return sample


def test_samples_with_different_attribute_counts_are_padded() -> None:
	batch = mae_collate_fn(
		[
			_sample((0, 2), use_context=False),
			_sample((0, 1, 2), use_context=False),
		],
	)

	assert batch['x'].shape == (2, 3, 2, 2, 2)
	assert torch.equal(batch['attribute_ids'][0], torch.tensor([0, 2, -1]))
	assert torch.equal(batch['attribute_ids'][1], torch.tensor([0, 1, 2]))
	assert not batch['x'][0, 2].any()


def test_padded_ids_are_negative_one_and_validity_mask_is_false() -> None:
	batch = mae_collate_fn(
		[
			_sample((0,), use_context=False),
			_sample((0, 1), use_context=False),
		],
	)

	assert batch['attribute_ids'].dtype == torch.long
	assert batch['attribute_ids'][0, 1].item() == -1
	assert torch.equal(
		batch['attribute_valid_mask'],
		torch.tensor([[True, False], [True, True]]),
	)


def test_context_is_collated_when_present() -> None:
	batch = mae_collate_fn(
		[
			_sample((0,), use_context=True),
			_sample((0, 1), use_context=True),
		],
	)

	assert batch['context'].shape == (2, 2, 2, 2, 2)
	assert batch['context_valid_mask'].shape == (2, 2, 2, 2)
	assert batch['context_valid_mask'].dtype == torch.bool
	assert not batch['context'][0, 1].any()


def test_context_absence_is_handled_when_use_context_false() -> None:
	batch = mae_collate_fn(
		[
			_sample((0,), use_context=False),
			_sample((0, 1), use_context=False),
		],
	)

	assert batch['context'] is None
	assert batch['context_valid_mask'] is None


def test_collated_none_context_batch_runs_through_model() -> None:
	batch = mae_collate_fn([_sample((0, 1), use_context=False)])
	batch['spatial_mask'] = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
	batch['visible_spatial_mask'] = ~batch['spatial_mask']
	model = StrictAttributeSetMAE3D(
		num_attributes=4,
		attribute_groups=None,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=16,
		encoder_depth=1,
		encoder_heads=4,
		decoder_dim=16,
		decoder_depth=1,
		decoder_heads=4,
		num_context_tokens=1,
		use_context=False,
	)

	out = model(batch)

	assert out['pred_patches'].shape == (1, 1, 4, 8)


def test_collated_dtypes_are_correct() -> None:
	batch = mae_collate_fn([_sample((0, 1), use_context=False)])

	assert batch['x'].dtype == torch.float32
	assert batch['target'].dtype == torch.float32
	assert batch['attribute_ids'].dtype == torch.long
	assert batch['spatial_mask'].dtype == torch.bool
	assert batch['visible_spatial_mask'].dtype == torch.bool
	assert batch['attribute_input_mask'].dtype == torch.bool
	assert batch['attribute_target_mask'].dtype == torch.bool
	assert batch['dropped_attribute_mask'].dtype == torch.bool
	assert batch['target_valid'].dtype == torch.bool


def test_move_batch_to_device_moves_tensors_and_preserves_coords() -> None:
	coords = {'survey_id': 'survey-a'}
	batch = mae_collate_fn([_sample((0, 1), use_context=False, coords=coords)])
	device = torch.device('cpu')

	moved = move_batch_to_device(batch, device)

	assert moved['x'].device == device
	assert moved['target'].device == device
	assert moved['coords'] == [coords]
	assert moved['coords'] is batch['coords']


def test_build_mae_dataloader_uses_mae_collate_fn() -> None:
	dataloader = build_mae_dataloader(
		[
			_sample((0,), use_context=False),
			_sample((0, 1), use_context=False),
		],
		batch_size=2,
		shuffle=False,
	)

	batch = next(iter(dataloader))

	assert batch['x'].shape == (2, 2, 2, 2, 2)
	assert torch.equal(
		batch['attribute_valid_mask'],
		torch.tensor([[True, False], [True, True]]),
	)
