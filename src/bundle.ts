import { featureStride } from 'webgpu-gaussian-rasterizer';

/**
 * Reader and writer for GSB1, the avatar bundle: the Gaussians of one
 * GAGAvatar avatar at its neutral head, and the neutral head vertices the
 * motion is relative to.
 *
 * Layout, little-endian: the bytes "GSB1", a u32 header length, 8 zero
 * bytes, the JSON header, then the arrays. The header lists each array's
 * name, dtype, shape, byte offset (from the end of the header) and byte
 * length. Array offsets are multiples of 16, and so is the end of the
 * header, so every array can be viewed in place.
 */

export const BUNDLE_MAGIC = 'GSB1';
const PREFIX_BYTES = 16;
const ALIGN = 16;

export type BundleDType = 'float32' | 'float16' | 'uint32' | 'int32' | 'uint8';
type TypedArray = Float32Array | Uint16Array | Uint32Array | Int32Array | Uint8Array;

const CTORS = {
  float32: Float32Array, float16: Uint16Array, uint32: Uint32Array, int32: Int32Array, uint8: Uint8Array,
} as const satisfies Record<BundleDType, { new (buffer: ArrayBufferLike, offset: number, length: number): TypedArray; BYTES_PER_ELEMENT: number }>;

export interface BundleArray {
  readonly dtype: BundleDType;
  readonly shape: readonly number[];
  /** float16 arrays hold their half-precision bits. */
  readonly data: TypedArray;
}

/** Where the head turns on the neck, in the avatar's world frame; see `createHeadTurnPass`. */
export interface NeckParameters {
  readonly pivot: readonly [number, number, number];
  readonly baseY: number;
  readonly topY: number;
}

export interface BundleMeta {
  /** Gaussians in the bundle. */
  count: number;
  /** The first `head_count` Gaussians sit on the head vertices and move with them. */
  head_count: number;
  /** tan of half the vertical field of view the avatar was reconstructed for. */
  tan_fov: number;
  feature_channels: number;
  /** The neck, for turning the head with the body kept still. */
  neck?: { pivot: [number, number, number]; base_y: number; top_y: number };
  /** The avatar's own camera, a 3x4 row-major transform as GAGAvatar builds its view from. */
  camera?: number[];
  [key: string]: unknown;
}

export interface AvatarBundle {
  readonly meta: BundleMeta;
  readonly arrays: Readonly<Record<string, BundleArray>>;
}

interface HeaderEntry { name: string; dtype: BundleDType; shape: number[]; offset: number; nbytes: number }

export function parseBundle(buffer: ArrayBuffer): AvatarBundle {
  const u8 = new Uint8Array(buffer);
  const magic = String.fromCharCode(...u8.subarray(0, 4));
  if (magic !== BUNDLE_MAGIC) throw new Error(`not a ${BUNDLE_MAGIC} bundle (magic ${JSON.stringify(magic)})`);
  const headerBytes = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(PREFIX_BYTES, PREFIX_BYTES + headerBytes))) as { meta: BundleMeta; arrays: HeaderEntry[] };
  const dataStart = PREFIX_BYTES + headerBytes;
  const arrays: Record<string, BundleArray> = {};
  for (const a of header.arrays) {
    const C = CTORS[a.dtype];
    if (!C) throw new Error(`array ${a.name} has unsupported dtype ${a.dtype}`);
    const start = dataStart + a.offset;
    if (start + a.nbytes > buffer.byteLength) throw new Error(`array ${a.name} runs past the end of the bundle`);
    const length = a.nbytes / C.BYTES_PER_ELEMENT;
    if (a.shape.reduce((x, y) => x * y, 1) !== length) throw new Error(`array ${a.name} has ${length} elements for shape [${a.shape}]`);
    // A writer that did not align its arrays still gets read, at the cost of a copy.
    const data = start % C.BYTES_PER_ELEMENT === 0 ? new C(buffer, start, length) : new C(buffer.slice(start, start + a.nbytes), 0, length);
    arrays[a.name] = { dtype: a.dtype, shape: a.shape, data };
  }
  return { meta: header.meta, arrays };
}

const pad = (n: number) => Math.ceil(n / ALIGN) * ALIGN;

/** Writes a bundle; the inverse of `parseBundle`, as `gagavatar_web.bundle.write_bundle` writes it in Python. */
export function encodeBundle(meta: BundleMeta, arrays: Readonly<Record<string, BundleArray>>): ArrayBuffer {
  const entries: HeaderEntry[] = [];
  let offset = 0;
  for (const [name, a] of Object.entries(arrays)) {
    entries.push({ name, dtype: a.dtype, shape: [...a.shape], offset, nbytes: a.data.byteLength });
    offset = pad(offset + a.data.byteLength);
  }
  const json = new TextEncoder().encode(JSON.stringify({ meta, arrays: entries }));
  // Spaces after the JSON keep the arrays aligned and still parse.
  const headerBytes = pad(PREFIX_BYTES + json.length) - PREFIX_BYTES;
  const out = new Uint8Array(PREFIX_BYTES + headerBytes + offset);
  out.set(new TextEncoder().encode(BUNDLE_MAGIC), 0);
  new DataView(out.buffer).setUint32(4, headerBytes, true);
  out.fill(0x20, PREFIX_BYTES, PREFIX_BYTES + headerBytes);
  out.set(json, PREFIX_BYTES);
  for (const e of entries) {
    const a = arrays[e.name];
    out.set(new Uint8Array(a.data.buffer, a.data.byteOffset, a.data.byteLength), PREFIX_BYTES + headerBytes + e.offset);
  }
  return out.buffer;
}

/** An avatar's Gaussians, in the layouts webgpu-gaussian-rasterizer takes. */
export interface AvatarGaussians {
  readonly count: number;
  readonly headCount: number;
  readonly channels: number;
  /** x y z per Gaussian; the head's at the neutral head. */
  readonly positions: Float32Array;
  /** xx xy xz yy yz zz per Gaussian. */
  readonly covariances: Float32Array;
  readonly opacities: Float32Array;
  /** Half-precision features, `featureStride(channels)` per Gaussian, as `writeFeatures` takes them. */
  readonly features: Uint16Array;
  /** The neutral head vertices, x y z per vertex. */
  readonly headRef: Float32Array;
  readonly tanFov: number;
  readonly neck: NeckParameters | null;
}

function array(bundle: AvatarBundle, name: string, dtype: BundleDType, shape: readonly number[]): TypedArray {
  const a = bundle.arrays[name];
  if (!a) throw new Error(`the bundle has no array ${name}`);
  if (a.dtype !== dtype || a.shape.length !== shape.length || a.shape.some((d, i) => d !== shape[i])) {
    throw new Error(`array ${name} is ${a.dtype} [${a.shape}], expected ${dtype} [${shape}]`);
  }
  return a.data;
}

/** Checks a bundle's arrays against its header and returns them by role. */
export function avatarGaussians(bundle: AvatarBundle): AvatarGaussians {
  const { count: n, head_count: headCount, feature_channels: channels, tan_fov: tanFov, neck } = bundle.meta;
  // Features are named by their channel count, so a bundle can carry
  // several projections of them side by side.
  const raw = array(bundle, `feat${channels}`, 'float16', [n, channels]) as Uint16Array;
  const stride = featureStride(channels);
  let features = raw;
  if (stride !== channels) {
    features = new Uint16Array(n * stride);
    for (let i = 0; i < n; i++) features.set(raw.subarray(i * channels, (i + 1) * channels), i * stride);
  }
  return {
    count: n, headCount, channels, tanFov, features,
    positions: array(bundle, 'pos', 'float32', [n, 3]) as Float32Array,
    covariances: array(bundle, 'cov', 'float32', [n, 6]) as Float32Array,
    opacities: array(bundle, 'opacity', 'float32', [n]) as Float32Array,
    headRef: array(bundle, 'head_ref', 'float32', [headCount, 3]) as Float32Array,
    neck: neck ? { pivot: neck.pivot, baseY: neck.base_y, topY: neck.top_y } : null,
  };
}
