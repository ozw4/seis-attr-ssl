from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THREAD_LIMIT_ENV = {
	'OMP_NUM_THREADS': '1',
	'MKL_NUM_THREADS': '1',
}


def thread_limited_env(extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
	env = os.environ.copy()
	env.update(THREAD_LIMIT_ENV)
	if extra_env is not None:
		env.update(extra_env)
	return env


def run_proc(
	args: Sequence[str | Path],
	*,
	timeout: float = 30,
	extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
	return subprocess.run(  # noqa: S603
		[str(arg) for arg in args],
		check=False,
		capture_output=True,
		text=True,
		cwd=PROJECT_ROOT,
		env=thread_limited_env(extra_env),
		timeout=timeout,
	)


def run_python_proc(
	script_path: Path,
	*args: str | Path,
	timeout: float = 30,
	extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
	return run_proc(
		[sys.executable, PROJECT_ROOT / script_path, *args],
		timeout=timeout,
		extra_env=extra_env,
	)
