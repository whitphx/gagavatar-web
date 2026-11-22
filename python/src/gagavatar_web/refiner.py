"""Exports GAGAvatar's upsampler, the network that turns the splatted
feature image into the picture, for ``createRefiner`` in the browser."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

# Tolerances the browser's load-time check allows, from the precision of
# the graph's arithmetic.
TOLERANCE = {"float32": 2e-3, "float16": 0.05}
CHECK_SEED = 7
CHECK_SAMPLES = 64


def hash_fill(n: int, seed: int, bound: float) -> np.ndarray:
    """The same values as ``hashFill`` in the JavaScript package: ``n``
    float32 values in [-bound, bound) from an integer hash of the index."""
    mix = np.uint32((seed + 1) * 0x9E3779B9 & 0xFFFFFFFF)
    x = np.arange(n, dtype=np.uint32) ^ mix
    with np.errstate(over="ignore"):
        x ^= x >> np.uint32(16)
        x *= np.uint32(0x7FEB352D)
        x ^= x >> np.uint32(15)
        x *= np.uint32(0x846CA68B)
        x ^= x >> np.uint32(16)
    # The JavaScript side divides in float64 and rounds once to float32.
    return ((x.astype(np.float64) / 4294967296 * 2 - 1) * bound).astype(np.float32)


def _check_indices(n: int, count: int) -> list[int]:
    return sorted({int(v) for v in (hash_fill(count, CHECK_SEED + 1, 0.5) + 0.5) * (n - 1)})


def _deterministic(upsampler):
    import torch

    # GAGAvatar's StyleUNet adds fresh random noise on every call unless told
    # to use the noise stored with its weights. Exported as is, the graph
    # would draw new noise per frame, flicker, and fail its own check.
    if "randomize_noise" not in inspect.signature(upsampler.forward).parameters:
        return upsampler

    class Deterministic(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x):
            return self.module(x, randomize_noise=False)

    return Deterministic(upsampler)


def export_refiner(upsampler, out: str | Path, *, channels: int = 32, size: int = 512, precision: str = "float16", device: str | None = None) -> dict[str, Any]:
    """Writes ``<out>.onnx`` and ``<out>.json``, the graph and the spec
    ``createRefiner`` takes (dims, data type, and output values for a fixed
    input that the browser checks at load).

    ``upsampler`` is the network, typically ``GAGAvatarRuntime(...).model.upsampler``.
    A float16 graph is exported from a half-precision copy of the network,
    on CUDA when it is available, since PyTorch's CPU kernels lack half
    precision for some of its layers.
    """
    import torch

    if precision not in TOLERANCE:
        raise ValueError(f"precision must be one of {sorted(TOLERANCE)}")
    out = Path(out)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    module = _deterministic(upsampler).to(device).eval()
    dims_in = [1, channels, size, size]
    probe = torch.from_numpy(hash_fill(int(np.prod(dims_in)), CHECK_SEED, 1.0).reshape(dims_in)).to(device)
    with torch.no_grad():
        result = module(probe)
    dims_out = list(result.shape)
    reference = result.float().cpu().numpy().reshape(-1)
    dtype = torch.float16 if precision == "float16" else torch.float32
    kwargs: dict[str, Any] = {"input_names": ["features"], "output_names": ["image"], "opset_version": 17}
    # The TorchScript exporter, which newer PyTorch keeps behind dynamo=False.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(module.to(dtype), (probe.to(dtype),), str(out.with_suffix(".onnx")), **kwargs)
    # Module.to converts in place, so the caller gets its network back as it was.
    module.float()
    indices = _check_indices(reference.size, CHECK_SAMPLES)
    spec = {
        "dataType": precision,
        "input": {"name": "features", "dims": dims_in},
        "output": {"name": "image", "dims": dims_out},
        "check": {
            "inputSeed": CHECK_SEED, "bound": 1.0, "indices": indices,
            "values": [float(reference[i]) for i in indices], "tolerance": TOLERANCE[precision],
        },
    }
    out.with_suffix(".json").write_text(json.dumps(spec, indent=1))
    return spec
