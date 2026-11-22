"""Reader and writer for GSB1 avatar bundles.

The layout is described in the JavaScript reader, packages/gagavatar-web/src/bundle.ts:
the bytes ``GSB1``, a u32 header length, 8 zero bytes, a JSON header, then
the arrays, each starting on a 16-byte boundary.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"GSB1"
PREFIX = struct.Struct("<4sI8x")
ALIGN = 16
DTYPES = {
    np.dtype("<f4"): "float32",
    np.dtype("<f2"): "float16",
    np.dtype("<u4"): "uint32",
    np.dtype("<i4"): "int32",
    np.dtype("u1"): "uint8",
}


def _pad(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def encode_bundle(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    entries, chunks, offset = [], [], 0
    for name, array in arrays.items():
        a = np.ascontiguousarray(array)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        if a.dtype not in DTYPES:
            raise TypeError(f"array {name} has unsupported dtype {a.dtype}")
        entries.append({"name": name, "dtype": DTYPES[a.dtype], "shape": list(a.shape), "offset": offset, "nbytes": a.nbytes})
        chunks.append((offset, a.tobytes()))
        offset = _pad(offset + a.nbytes)
    header = json.dumps({"meta": meta, "arrays": entries}).encode()
    # Spaces after the JSON keep the arrays aligned and still parse.
    header_bytes = _pad(PREFIX.size + len(header)) - PREFIX.size
    out = bytearray(PREFIX.size + header_bytes + offset)
    PREFIX.pack_into(out, 0, MAGIC, header_bytes)
    out[PREFIX.size : PREFIX.size + header_bytes] = header.ljust(header_bytes, b" ")
    start = PREFIX.size + header_bytes
    for at, data in chunks:
        out[start + at : start + at + len(data)] = data
    return bytes(out)


def write_bundle(path: str | Path, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    Path(path).write_bytes(encode_bundle(meta, arrays))


def decode_bundle(data: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    magic, header_bytes = PREFIX.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"not a GSB1 bundle (magic {magic!r})")
    header = json.loads(data[PREFIX.size : PREFIX.size + header_bytes])
    start = PREFIX.size + header_bytes
    names = {v: k for k, v in DTYPES.items()}
    arrays = {}
    for e in header["arrays"]:
        a = np.frombuffer(data, dtype=names[e["dtype"]], count=int(np.prod(e["shape"])), offset=start + e["offset"])
        arrays[e["name"]] = a.reshape(e["shape"])
    return header["meta"], arrays


def read_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return decode_bundle(Path(path).read_bytes())
