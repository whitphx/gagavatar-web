"""Exports one GAGAvatar avatar as a GSB1 bundle for the browser renderer.

Runs where GAGAvatar is installed, with its weights, the FLAME model and a
tracked avatar of the user's own. The bundle holds GAGAvatar's Gaussians
for that avatar at its neutral head, the neutral head vertices, and the
neck the browser turns the head on. All three derive from the FLAME model
the user supplies, so the bundle comes under FLAME's licence too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bundle import write_bundle
from .covariance import covariances_from_scale_rotation

HEAD_VERTICES = 5023
# FLAME's kinematic chain: the root sits at the base of the neck, the neck
# joint behind the head's centre.
FLAME_ROOT_JOINT = 0
FLAME_NECK_JOINT = 1
# The chin among FLAME's 68 landmarks (iBUG numbering, the bottom of the jaw line).
FLAME_CHIN_LANDMARK = 8


def neck_parameters(flame_model, shape_code) -> dict[str, Any]:
    """The pivot the head turns about and the heights the turn ramps in
    between, for one avatar's FLAME shape (``shape_code``, (1, n_shape)).

    The ramp ends at the chin, so that the whole face turns rigidly. FLAME's
    neck joint is no good as the top: it sits at the height of the nose, and
    a ramp up to it gives the chin a third of the turn and the lips two
    thirds, which shears the face whenever the head turns.
    """
    import torch

    with torch.no_grad():
        shape = shape_code.to(flame_model.v_template.device)
        v = flame_model.v_template + torch.einsum("bl,mkl->bmk", shape, flame_model.shapedirs[:, :, : shape.shape[-1]])
        v = v[0] * flame_model.scale
        joints = flame_model.J_regressor @ v
        face = flame_model.faces_tensor[flame_model.full_lmk_faces_idx[0, FLAME_CHIN_LANDMARK]].long()
        chin = (v[face] * flame_model.full_lmk_bary_coords[0, FLAME_CHIN_LANDMARK][:, None]).sum(0)
    root, neck = joints[FLAME_ROOT_JOINT], joints[FLAME_NECK_JOINT]
    return {
        "pivot": [float(c) for c in (root + neck) * 0.5],
        "base_y": float(root[1]),
        "top_y": float(chin[1]),
    }


def export_bundle(
    runtime,
    out: str | Path,
    *,
    prune: float = 0.01,
    projection: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Writes the bundle of the avatar ``runtime`` has selected and returns its meta.

    ``runtime`` is a ``gagavatar.runtime.GAGAvatarRuntime`` after
    ``set_avatar_id`` or ``set_tracked_avatar``. Gaussians off the head
    whose opacity is under ``prune`` are left out. ``projection``, a
    (mean (C,), basis (C, K)) pair, stores ``(features - mean) @ basis`` in
    place of GAGAvatar's C = 32 raw feature channels, for a refiner trained
    on K projected channels.
    """
    import torch

    with torch.no_grad():
        batch = runtime.build_forward_batch(torch.zeros(1, 106))
        gs = runtime.model.forward_gaussians(batch)
    xyz = gs["xyz"][0].float().cpu().numpy()
    colors = gs["colors"][0].float().cpu().numpy()
    opacity = gs["opacities"][0, :, 0].float().cpu().numpy()
    scales = gs["scales"][0].float().cpu().numpy()
    rotations = gs["rotations"][0].float().cpu().numpy()
    keep = opacity >= prune
    keep[:HEAD_VERTICES] = True
    features = colors[keep]
    if projection is not None:
        mean, basis = projection
        features = (features - mean) @ basis
    camera = batch["t_transform"][0, :3, :4].float().cpu().numpy()
    meta = {
        "count": int(keep.sum()),
        "head_count": HEAD_VERTICES,
        "tan_fov": 1.0 / float(runtime.model.cam_params["focal_y"]),
        "feature_channels": int(features.shape[1]),
        "neck": neck_parameters(runtime.flame_model, runtime.shape_code()),
        "camera": [float(v) for v in camera.reshape(-1)],
        "prune": prune,
    }
    write_bundle(out, meta, {
        "pos": xyz[keep].astype(np.float32),
        "cov": covariances_from_scale_rotation(scales[keep], rotations[keep]),
        "opacity": opacity[keep].astype(np.float32),
        f"feat{features.shape[1]}": features.astype(np.float16),
        "head_ref": xyz[:HEAD_VERTICES].astype(np.float32),
    })
    return meta
