"""Thin entrypoint for amplitude-only MAE training."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_ssl_cluster.utils.cli import run_pending_entrypoint  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'train_amp_mae.yaml'
)


def main() -> None:
	"""Validate config and report pending amplitude-only MAE training."""
	run_pending_entrypoint(
		'Train an amplitude-only MAE model.',
		DEFAULT_CONFIG,
	)


if __name__ == '__main__':
	main()
