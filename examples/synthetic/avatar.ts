// A made-up avatar for the example and the tests: a sphere of Gaussians for
// the head, a mouth that opens, and a still neck and shoulders below. Its
// frame and camera follow GAGAvatar's, so the library treats it as it would
// a real bundle. Nothing in it comes from FLAME or GAGAvatar.

import { covariancesFromScaleRotation, toHalf } from 'webgpu-gaussian-rasterizer';
import { rotationMatrix, type AvatarBundle, type BundleMeta } from '../../src/index.js';

// GAGAvatar's camera distance (5000 / 512) and focal length (12, in units
// of half the image), which put a unit-sized head in frame.
const CAMERA_DISTANCE = 5000 / 512;
const TAN_FOV = 1 / 12;
const HEAD_RADIUS = 0.5;
const NECK = { pivot: [0, -0.62, 0] as [number, number, number], base_y: -0.85, top_y: -0.45 };

export interface SyntheticAvatar {
  bundle: AvatarBundle;
  /** The head vertices with the mouth `open` (0 to 1) of the way open. */
  vertices(open: number, out?: Float32Array): Float32Array;
  /** GAGAvatar's camera with the head turned by `yaw` and `pitch` radians. */
  camera(yaw: number, pitch: number, out?: Float32Array): Float32Array;
}

function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function createSyntheticAvatar({ rings = 24, segments = 32, bodyCount = 6000, channels = 3, seed = 1 } = {}): SyntheticAvatar {
  const random = mulberry32(seed);
  const head: number[] = [];
  for (let r = 0; r < rings; r++) {
    const theta = Math.PI * (r + 0.5) / rings;
    for (let s = 0; s < segments; s++) {
      const phi = 2 * Math.PI * s / segments;
      head.push(HEAD_RADIUS * Math.sin(theta) * Math.sin(phi), HEAD_RADIUS * Math.cos(theta), HEAD_RADIUS * Math.sin(theta) * Math.cos(phi));
    }
  }
  const headCount = head.length / 3;
  const headRef = new Float32Array(head);
  // The mouth: front vertices in a band below the head's middle.
  const mouth = new Float32Array(headCount);
  for (let i = 0; i < headCount; i++) {
    const [x, y, z] = [headRef[3 * i], headRef[3 * i + 1], headRef[3 * i + 2]];
    mouth[i] = z > 0.25 && y < -0.05 ? Math.max(0, 1 - Math.hypot(x / 0.22, (y + 0.22) / 0.14)) : 0;
  }

  const count = headCount + bodyCount;
  const pos = new Float32Array(3 * count);
  pos.set(headRef);
  for (let i = headCount; i < count; i++) {
    // Half of the rest is the neck, a cylinder; the other half the shoulders, a flattened dome.
    if (i % 2) {
      const a = 2 * Math.PI * random(), y = -0.95 + 0.6 * random();
      pos.set([0.22 * Math.sin(a), y, 0.22 * Math.cos(a) - 0.05], 3 * i);
    } else {
      const a = Math.PI * (random() - 0.5) * 1.6, r = Math.sqrt(random());
      pos.set([1.1 * r * Math.sin(a) * 1.4, -0.85 - 0.35 * random(), 0.35 * r * Math.cos(a) - 0.1], 3 * i);
    }
  }
  const scales = new Float32Array(3 * count), rotations = new Float32Array(4 * count);
  const colours = new Float32Array(channels * count), opacity = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const s = i < headCount ? 0.045 : 0.03 + 0.03 * random();
    scales.set([s, s * (0.6 + 0.4 * random()), s], 3 * i);
    const q = [1 + random(), random() - 0.5, random() - 0.5, random() - 0.5];
    rotations.set(q.map((v) => v / Math.hypot(...q)), 4 * i);
    opacity[i] = 0.6 + 0.35 * random();
    const y = pos[3 * i + 1];
    const rgb = i < headCount
      ? (mouth[i] > 0 ? [0.55, 0.12, 0.15] : [0.93, 0.76 + 0.1 * y, 0.64])
      : (y > -0.8 ? [0.9, 0.72, 0.6] : [0.18, 0.32, 0.6]);
    for (let c = 0; c < channels; c++) colours[channels * i + c] = c < 3 ? rgb[c] * (0.92 + 0.08 * random()) : random();
  }
  const meta: BundleMeta = { count, head_count: headCount, tan_fov: TAN_FOV, feature_channels: channels, neck: NECK };
  const bundle: AvatarBundle = {
    meta,
    arrays: {
      pos: { dtype: 'float32', shape: [count, 3], data: pos },
      cov: { dtype: 'float32', shape: [count, 6], data: covariancesFromScaleRotation(scales, rotations) },
      opacity: { dtype: 'float32', shape: [count], data: opacity },
      [`feat${channels}`]: { dtype: 'float16', shape: [count, channels], data: Uint16Array.from(colours, toHalf) },
      head_ref: { dtype: 'float32', shape: [headCount, 3], data: headRef },
    },
  };

  return {
    bundle,
    vertices(open, out = new Float32Array(3 * headCount)) {
      out.set(headRef);
      for (let i = 0; i < headCount; i++) out[3 * i + 1] -= 0.12 * open * mouth[i];
      return out;
    },
    camera(yaw, pitch, out = new Float32Array(12)) {
      // R = H^T R_ref for a head turned by H (see takeHeadTurnOffCamera),
      // with R_ref = diag(-1, 1, -1).
      const h = rotationMatrix(axisAngle(yaw, pitch));
      const ref = [-1, 1, -1];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) out[i * 4 + j] = h[j * 3 + i] * ref[j];
      }
      out[3] = 0; out[7] = 0; out[11] = CAMERA_DISTANCE;
      return out;
    },
  };
}

// The rotation by `pitch` about x after `yaw` about y, as one axis-angle vector.
function axisAngle(yaw: number, pitch: number): [number, number, number] {
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2), cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
  // q = q_pitch * q_yaw, (w, x, y, z).
  const w = cp * cy, x = sp * cy, y = cp * sy, z = sp * sy;
  const s = Math.hypot(x, y, z);
  if (s < 1e-12) return [0, 0, 0];
  const angle = 2 * Math.atan2(s, w);
  return [x / s * angle, y / s * angle, z / s * angle];
}
