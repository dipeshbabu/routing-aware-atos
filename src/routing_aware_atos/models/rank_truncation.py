
from __future__ import annotations

import numpy as np


def truncate_matrix_rank(matrix: np.ndarray, rank: int | None) -> np.ndarray:
    """Return a rank-truncated copy of the matrix using SVD."""
    if rank is None:
        return matrix.copy()
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    max_rank = min(matrix.shape)
    if rank >= max_rank:
        return matrix.copy()

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]
