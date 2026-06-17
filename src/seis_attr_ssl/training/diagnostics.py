"""Diagnostic payload helpers for MAE training failures."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


def summarize_tensor(value: object) -> dict[str, object]:  # noqa: PLR0911
	"""Return a compact JSON-safe summary for ``value``."""
	if value is None:
		return {'present': False}
	if not isinstance(value, torch.Tensor):
		return {
			'present': True,
			'type': type(value).__name__,
			'repr': repr(value),
		}

	summary: dict[str, object] = {
		'present': True,
		'dtype': str(value.dtype),
		'shape': [int(dim) for dim in value.shape],
		'numel': int(value.numel()),
	}
	if value.numel() == 0:
		return summary

	detached = value.detach()
	if detached.dtype == torch.bool:
		true_count = int(detached.sum().cpu().item())
		summary['true_count'] = true_count
		summary['false_count'] = int(detached.numel() - true_count)
		return summary

	if torch.is_floating_point(detached):
		return _summarize_float_tensor(detached, summary)

	if detached.is_complex():
		summary['finite_count'] = int(torch.isfinite(detached).sum().cpu().item())
		summary['all_finite'] = bool(torch.isfinite(detached).all().cpu().item())
		return summary

	cpu = detached.cpu()
	summary['min'] = _json_safe_number(cpu.min().item())
	summary['max'] = _json_safe_number(cpu.max().item())
	return summary


def summarize_loss_components(
	losses: Mapping[str, torch.Tensor],
) -> dict[str, object]:
	"""Return JSON-safe scalar summaries for loss components."""
	return {key: _summarize_loss_value(value) for key, value in losses.items()}


def build_mae_nonfinite_diagnostic(  # noqa: PLR0913
	*,
	global_step: int,
	epoch: int,
	batch_index: int,
	batch: Mapping[str, object],
	output: Mapping[str, object],
	losses: Mapping[str, torch.Tensor],
	torch_amp_enabled: bool | None = None,
) -> dict[str, object]:
	"""Build the diagnostic payload for a non-finite MAE loss."""
	coords = _json_safe(batch.get('coords'))
	coord_items = coords if isinstance(coords, list) else []
	return {
		'global_step': int(global_step),
		'epoch': int(epoch),
		'batch_index': int(batch_index),
		'survey_id': _coord_values(coord_items, 'survey_id'),
		'local_start_xyz': _coord_values(coord_items, 'local_start_xyz'),
		'local_compute_start_xyz': _coord_values(
			coord_items,
			'local_compute_start_xyz',
		),
		'context_compute_start_xyz': _coord_values(
			coord_items,
			'context_compute_start_xyz',
		),
		'coords': coords,
		'attribute_ids': _json_safe_tensor(batch.get('attribute_ids')),
		'losses': summarize_loss_components(losses),
		'tensors': {
			'x': summarize_tensor(batch.get('x')),
			'target': summarize_tensor(batch.get('target')),
			'context': summarize_tensor(batch.get('context')),
			'pred_patches': summarize_tensor(output.get('pred_patches')),
			'target_valid': summarize_tensor(batch.get('target_valid')),
			'spatial_mask': summarize_tensor(batch.get('spatial_mask')),
			'dropped_attribute_mask': summarize_tensor(
				batch.get('dropped_attribute_mask'),
			),
			'context_valid_mask': summarize_tensor(batch.get('context_valid_mask')),
		},
		'torch_amp_enabled': (
			bool(torch.is_autocast_enabled('cuda'))
			if torch_amp_enabled is None
			else bool(torch_amp_enabled)
		),
	}


def write_json_diagnostic(payload: Mapping[str, object], path: str | Path) -> Path:
	"""Write ``payload`` as strict JSON and return the output path."""
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
	output_path.write_text(f'{text}\n', encoding='utf-8')
	return output_path


def _summarize_float_tensor(
	value: torch.Tensor,
	summary: dict[str, object],
) -> dict[str, object]:
	finite = torch.isfinite(value)
	finite_count = int(finite.sum().cpu().item())
	nan_count = int(torch.isnan(value).sum().cpu().item())
	posinf_count = int(torch.isposinf(value).sum().cpu().item())
	neginf_count = int(torch.isneginf(value).sum().cpu().item())
	summary.update(
		{
			'finite_count': finite_count,
			'nan_count': nan_count,
			'posinf_count': posinf_count,
			'neginf_count': neginf_count,
			'all_finite': finite_count == value.numel(),
		},
	)
	if finite_count == 0:
		summary.update({'min': None, 'max': None, 'mean': None})
		return summary

	finite_values = value[finite].float().cpu()
	summary.update(
		{
			'min': _json_safe_number(finite_values.min().item()),
			'max': _json_safe_number(finite_values.max().item()),
			'mean': _json_safe_number(finite_values.mean().item()),
		},
	)
	return summary


def _summarize_loss_value(value: torch.Tensor) -> dict[str, object]:
	if not isinstance(value, torch.Tensor):
		return {'present': True, 'type': type(value).__name__, 'repr': repr(value)}
	if value.numel() != 1:
		return summarize_tensor(value)
	item = value.detach().cpu().item()
	if isinstance(item, float) and not math.isfinite(item):
		return {'value': None, 'finite': False, 'repr': repr(item)}
	return {'value': _json_safe_number(item), 'finite': True}


def _json_safe_number(value: object) -> object:
	if isinstance(value, bool):
		return value
	if isinstance(value, int):
		return int(value)
	if isinstance(value, float):
		if math.isfinite(value):
			return float(value)
		return {'value': None, 'finite': False, 'repr': repr(value)}
	return value


def _json_safe_tensor(value: object) -> object:
	if not isinstance(value, torch.Tensor):
		return _json_safe(value)
	if value.numel() > 4096:
		return summarize_tensor(value)
	return _json_safe(value.detach().cpu().tolist())


def _coord_values(coords: Sequence[object], key: str) -> list[object]:
	values: list[object] = []
	for coord in coords:
		if isinstance(coord, Mapping):
			values.append(_json_safe(coord.get(key)))
		else:
			values.append(None)
	return values


def _json_safe(value: object) -> object:  # noqa: PLR0911
	if isinstance(value, torch.Tensor):
		return _json_safe_tensor(value)
	if isinstance(value, Mapping):
		return {str(key): _json_safe(child) for key, child in value.items()}
	if isinstance(value, tuple | list):
		return [_json_safe(child) for child in value]
	if isinstance(value, bool | str) or value is None:
		return value
	if isinstance(value, int):
		return int(value)
	if isinstance(value, float):
		if math.isfinite(value):
			return float(value)
		return {'value': None, 'finite': False, 'repr': repr(value)}
	return repr(value)


__all__ = [
	'build_mae_nonfinite_diagnostic',
	'summarize_loss_components',
	'summarize_tensor',
	'write_json_diagnostic',
]
