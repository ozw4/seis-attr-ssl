from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROC_SCRIPTS = (
	Path('proc/build_nopims_manifests.py'),
	Path('proc/generate_attributes.py'),
	Path('proc/train_mae.py'),
	Path('proc/train_dense_adapt.py'),
	Path('proc/train_finetune.py'),
	Path('proc/eval_f3.py'),
	Path('proc/infer_volume.py'),
)


@pytest.mark.parametrize('script_path', PROC_SCRIPTS)
def test_proc_script_help_exits_zero(script_path: Path) -> None:
	result = subprocess.run(  # noqa: S603
		[sys.executable, str(PROJECT_ROOT / script_path), '--help'],
		check=False,
		capture_output=True,
		text=True,
		cwd=PROJECT_ROOT,
	)

	assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('script_path', PROC_SCRIPTS)
def test_proc_script_dry_run_exits_zero_and_prints_stage(
	script_path: Path,
) -> None:
	result = subprocess.run(  # noqa: S603
		[sys.executable, str(PROJECT_ROOT / script_path), '--dry-run'],
		check=False,
		capture_output=True,
		text=True,
		cwd=PROJECT_ROOT,
	)

	assert result.returncode == 0, result.stderr
	assert 'stage:' in result.stdout


def test_train_mae_dry_run_prints_masking_settings() -> None:
	result = subprocess.run(  # noqa: S603
		[sys.executable, str(PROJECT_ROOT / 'proc/train_mae.py'), '--dry-run'],
		check=False,
		capture_output=True,
		text=True,
		cwd=PROJECT_ROOT,
	)

	assert result.returncode == 0, result.stderr
	assert 'masking.spatial_mask_ratio: 0.75' in result.stdout
	assert 'masking.spatial_mask_mode: block' in result.stdout
	assert 'masking.block_size_tokens: 2, 2, 2' in result.stdout
	assert 'train.samples_per_epoch: 10000' in result.stdout
	assert 'train.num_workers: 4' in result.stdout
	assert 'train.shuffle: true' in result.stdout


@pytest.mark.parametrize(
	('args', 'expected_stderr'),
	[
		(('--max-steps', '-1'), 'train.max_steps must be positive'),
		(('--output-root', 'F3/run'), 'F3 paths are not allowed'),
	],
)
def test_train_mae_dry_run_validates_cli_overrides(
	args: tuple[str, ...],
	expected_stderr: str,
) -> None:
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			str(PROJECT_ROOT / 'proc/train_mae.py'),
			'--dry-run',
			*args,
		],
		check=False,
		capture_output=True,
		text=True,
		cwd=PROJECT_ROOT,
	)

	assert result.returncode != 0
	assert expected_stderr in result.stderr
	assert 'stage:' not in result.stdout
