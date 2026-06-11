from __future__ import annotations

import os
from contextlib import suppress

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

try:
	import torch
except ImportError:
	torch = None

if torch is not None:
	torch.set_num_threads(1)
	with suppress(RuntimeError):
		torch.set_num_interop_threads(1)
