from __future__ import annotations

import numpy as np


def covariances_from_scale_rotation(scales: np.ndarray, rotations: np.ndarray, normalize: bool = False) -> np.ndarray:
    """3D covariances (N, 6) as xx, xy, xz, yy, yz, zz from scales (N, 3)
    and real-first quaternions (N, 4).

    The CUDA rasterizer GAGAvatar is trained against builds the rotation from
    the quaternion as given, so ``normalize`` defaults to False to reproduce
    it for quaternions that are not unit length. The JavaScript
    ``covariancesFromScaleRotation`` of webgpu-gaussian-rasterizer computes
    the same.
    """
    q = np.asarray(rotations, dtype=np.float64)
    if normalize:
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    r = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    m = r * np.asarray(scales, dtype=np.float64)[:, None, :]
    sigma = m @ m.transpose(0, 2, 1)
    return sigma[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].astype(np.float32)
