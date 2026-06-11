"""Entrypoint for MAE pretraining."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.config import load_config, validate_config  # noqa: E402
from seis_attr_ssl.training.mae import run_mae_pretraining  # noqa: E402
from seis_attr_ssl.utils.cli import (  # noqa: E402
	parse_config_args,
	print_config_summary,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'mvp_mae.yaml'


def main() -> None:
	"""Run MAE pretraining or print a dry-run config summary."""
	args = parse_config_args(
		'Pretrain the strict-attribute-set MAE model.',
		DEFAULT_CONFIG,
	)
	config = validate_config(load_config(args.config))
	if args.dry_run:
		print_config_summary(config)
		return

	checkpoint_path = run_mae_pretraining(config)
	print(f'checkpoint: {checkpoint_path}')


if __name__ == '__main__':
	main()
