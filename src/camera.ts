import type { Camera } from 'webgpu-gaussian-rasterizer';

/**
 * GAGAvatar's camera is a 3x4 row-major world-to-view transform [R | T] in
 * PyTorch3D's convention: row vectors (x_view = x R + T), x left, y up and
 * z forward. The rasterizer takes OpenCV's axes (x right, y down), which
 * negates the first two view axes, and a column-major matrix for column
 * vectors. That matrix's column-major storage is the row-vector matrix
 * [[R, 0], [T, 1]] stored row by row, so the copy below needs no transpose.
 */
export function viewFromCamera(camera: ArrayLike<number>, out = new Float32Array(16)): Float32Array {
  if (camera.length !== 12) throw new Error(`a camera is 3x4, got ${camera.length} values`);
  out.fill(0);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) out[i * 4 + j] = camera[i * 4 + j];
    out[12 + i] = camera[i * 4 + 3];
  }
  out[15] = 1;
  for (let i = 0; i < 4; i++) {
    out[i * 4] = -out[i * 4];
    out[i * 4 + 1] = -out[i * 4 + 1];
  }
  return out;
}

/** The rasterizer's camera for a GAGAvatar camera, square images of `size` pixels and the bundle's `tan_fov`. */
export function rasterizerCamera(camera: ArrayLike<number>, size: number, tanFov: number, view = new Float32Array(16)): Camera {
  const f = size / (2 * tanFov);
  return { view: viewFromCamera(camera, view), fx: f, fy: f };
}
