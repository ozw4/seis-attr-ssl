from __future__ import annotations

import pytest

from seis_ssl_cluster.embedding.sliding_window import (
	iter_sliding_windows,
	padded_volume_shape_xyz,
	token_grid_shape_xyz,
)


def test_sliding_windows_cover_patch_padded_volume() -> None:
	windows = list(
		iter_sliding_windows(
			(5, 6, 7),
			(4, 4, 4),
			(2, 2, 2),
			(2, 2, 2),
		),
	)

	assert padded_volume_shape_xyz((5, 6, 7), (2, 2, 2)) == (6, 6, 8)
	assert token_grid_shape_xyz((5, 6, 7), (2, 2, 2)) == (3, 3, 4)
	assert windows[0].start_xyz == (0, 0, 0)
	assert windows[-1].start_xyz == (2, 2, 4)


def test_sliding_windows_reject_non_patch_aligned_overlap() -> None:
	with pytest.raises(ValueError, match=r'overlap_xyz.*patch_size_xyz'):
		list(iter_sliding_windows((8, 8, 8), (4, 4, 4), (1, 0, 0), (2, 2, 2)))
