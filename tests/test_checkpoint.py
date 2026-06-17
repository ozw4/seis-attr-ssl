from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from seis_attr_ssl.training import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
	from pathlib import Path


def test_save_checkpoint_defaults_and_plain_config_values(tmp_path: Path) -> None:
	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	checkpoint_path = save_checkpoint(
		tmp_path / 'checkpoint.pt',
		model=model,
		optimizer=optimizer,
		epoch=2,
		config={
			'path': tmp_path / 'data',
			'nested': {'weights': tmp_path / 'weights.pt'},
		},
		metrics={'loss': 1.25},
		training_state={
			'schema_version': 1,
			'nested': {'artifact': tmp_path / 'artifact.json'},
		},
	)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	assert payload['epoch'] == 2
	assert payload['global_step'] == 0
	assert payload['amp_enabled'] is False
	assert payload['scaler_state_dict'] is None
	assert payload['config']['path'] == str(tmp_path / 'data')
	assert payload['config']['nested']['weights'] == str(tmp_path / 'weights.pt')
	assert payload['training_state'] == {
		'schema_version': 1,
		'nested': {'artifact': str(tmp_path / 'artifact.json')},
	}
	assert payload['metrics'] == {'loss': 1.25}


def test_load_checkpoint_accepts_old_minimal_payload(tmp_path: Path) -> None:
	checkpoint_path = tmp_path / 'old.pt'
	old_payload = {
		'model_state_dict': {'weight': torch.ones(1)},
		'optimizer_state_dict': {},
		'epoch': 1,
		'config': {'stage': 'pretrain_mae'},
		'package_version': None,
		'metrics': {'loss': 0.0},
	}
	torch.save(old_payload, checkpoint_path)

	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	assert torch.equal(
		payload['model_state_dict']['weight'],
		old_payload['model_state_dict']['weight'],
	)
	assert payload['optimizer_state_dict'] == {}
	assert payload['epoch'] == 1
	assert payload['config'] == {'stage': 'pretrain_mae'}
	assert payload['package_version'] is None
	assert payload['metrics'] == {'loss': 0.0}
