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
