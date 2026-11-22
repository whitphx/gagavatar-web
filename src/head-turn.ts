import type { NeckParameters } from './bundle.js';

/**
 * GAGAvatar applies the head pose as a camera rotation, which turns the
 * whole reconstructed bust, shoulders included, about the centre of the
 * head. To keep the body still, the camera goes back to the zero-pose view
 * and the head pose turns the Gaussians instead, each by a fraction that
 * ramps along the neck: none below the base of the neck, all of it from
 * the chin up, so the face turns rigidly and the neck bends.
 */

// GAGAvatar's camera rotation at zero head pose (transform_emoca_to_p3d of
// a zero rotation), which is diagonal.
const REFERENCE_DIAGONAL = [-1, 1, -1] as const;

/**
 * Splits the head pose off a GAGAvatar camera (3x4 row-major, rotation R
 * acting on row vectors). Writes the zero-pose camera, with the camera's
 * translation, into `outCamera` and returns the head turn as an axis-angle
 * vector, which turns a point taken as a column vector by M = R_ref R^T.
 * With that turn applied to the Gaussians, the zero-pose camera sees the
 * head as the original camera did.
 */
export function takeHeadTurnOffCamera(
  camera: ArrayLike<number>,
  outCamera: Float32Array = new Float32Array(12),
  outTurn: Float32Array = new Float32Array(3),
): Float32Array {
  // M[i][j] = d_i R[j][i], with R_ref = diag(d).
  const m = (i: number, j: number) => REFERENCE_DIAGONAL[i] * camera[j * 4 + i];
  const sx = (m(2, 1) - m(1, 2)) / 2, sy = (m(0, 2) - m(2, 0)) / 2, sz = (m(1, 0) - m(0, 1)) / 2;
  const sin = Math.hypot(sx, sy, sz), cos = (m(0, 0) + m(1, 1) + m(2, 2) - 1) / 2;
  // The skew part loses the axis only near a half turn, which no head pose
  // comes close to.
  const scale = sin > 1e-12 ? Math.atan2(sin, cos) / sin : 1;
  outTurn[0] = sx * scale; outTurn[1] = sy * scale; outTurn[2] = sz * scale;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) outCamera[i * 4 + j] = i === j ? REFERENCE_DIAGONAL[i] : 0;
    outCamera[i * 4 + 3] = camera[i * 4 + 3];
  }
  return outTurn;
}

/** Rodrigues' formula: the rotation by axis-angle `v`, as a row-major 3x3 for column vectors. */
export function rotationMatrix(v: ArrayLike<number>): number[] {
  const angle = Math.hypot(v[0], v[1], v[2]);
  if (angle < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const [x, y, z] = [v[0] / angle, v[1] / angle, v[2] / angle];
  const c = Math.cos(angle), s = Math.sin(angle), r = 1 - c;
  return [
    c + r * x * x, r * x * y - s * z, r * x * z + s * y,
    r * x * y + s * z, c + r * y * y, r * y * z - s * x,
    r * x * z - s * y, r * y * z + s * x, c + r * z * z,
  ];
}

/** The fraction of the turn a Gaussian at rest height `y` takes. */
export function neckWeight(neck: NeckParameters, y: number): number {
  return Math.min(1, Math.max(0, (y - neck.baseY) / (neck.topY - neck.baseY)));
}

const shader = /* wgsl */ `
struct Params {
  pivot: vec3f, base_y: f32,
  turn: vec3f, top_y: f32,
  count: u32, enabled: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src_pos: array<f32>;
@group(0) @binding(2) var<storage, read> src_cov: array<f32>;
@group(0) @binding(3) var<storage, read> rest_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_pos: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst_cov: array<f32>;

fn rotation(v: vec3f) -> mat3x3f {
  let angle = length(v);
  if (angle < 1e-12) { return mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0)); }
  let k = v / angle;
  let c = cos(angle);
  let s = sin(angle);
  let r = 1.0 - c;
  return mat3x3f(
    vec3f(c + r * k.x * k.x, s * k.z + r * k.x * k.y, -s * k.y + r * k.x * k.z),
    vec3f(-s * k.z + r * k.x * k.y, c + r * k.y * k.y, s * k.x + r * k.y * k.z),
    vec3f(s * k.y + r * k.x * k.z, -s * k.x + r * k.y * k.z, c + r * k.z * k.z));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) { return; }
  var p = vec3f(src_pos[3u * i], src_pos[3u * i + 1u], src_pos[3u * i + 2u]);
  var c = array<f32, 6>(src_cov[6u * i], src_cov[6u * i + 1u], src_cov[6u * i + 2u], src_cov[6u * i + 3u], src_cov[6u * i + 4u], src_cov[6u * i + 5u]);
  // Copied untouched when the turn is off, rather than turned by the
  // identity, whose round trip through the pivot would move every Gaussian
  // by a rounding error.
  if (P.enabled == 1u) {
    // The weight goes by the rest height, so an expression that moves a
    // vertex along the neck (the jaw opening) does not change its share of
    // the turn.
    let w = clamp((rest_y[i] - P.base_y) / (P.top_y - P.base_y), 0.0, 1.0);
    let M = rotation(w * P.turn);
    p = M * (p - P.pivot) + P.pivot;
    let S = M * mat3x3f(vec3f(c[0], c[1], c[2]), vec3f(c[1], c[3], c[4]), vec3f(c[2], c[4], c[5])) * transpose(M);
    c = array<f32, 6>(S[0][0], S[0][1], S[0][2], S[1][1], S[1][2], S[2][2]);
  }
  dst_pos[3u * i] = p.x; dst_pos[3u * i + 1u] = p.y; dst_pos[3u * i + 2u] = p.z;
  for (var k = 0u; k < 6u; k++) { dst_cov[6u * i + k] = c[k]; }
}`;

export interface HeadTurnPassOptions {
  count: number;
  neck: NeckParameters | null;
  /** Each Gaussian's height at rest, which sets its share of the turn. */
  restHeights: Float32Array;
  /** Posed positions and rest covariances, read every frame. */
  source: { positions: GPUBuffer; covariances: GPUBuffer };
  /** Where the turned Gaussians go: usually the rasterizer's buffers. */
  target: { positions: GPUBuffer; covariances: GPUBuffer };
}

export interface HeadTurnPass {
  /** The turn as an axis-angle vector (see `takeHeadTurnOffCamera`), or null for none. Applies from the next submit. */
  setTurn(turn: ArrayLike<number> | null): void;
  encode(encoder: GPUCommandEncoder): void;
  destroy(): void;
}

/** A compute pass that turns the head on its neck, writing the turned Gaussians into `target`. */
export function createHeadTurnPass(device: GPUDevice, options: HeadTurnPassOptions): HeadTurnPass {
  const { count, neck, restHeights, source, target } = options;
  if (restHeights.length !== count) throw new Error(`${restHeights.length} rest heights for ${count} Gaussians`);
  const module = device.createShaderModule({ code: shader, label: 'head_turn' });
  const pipeline = device.createComputePipeline({ label: 'head_turn', layout: 'auto', compute: { module, entryPoint: 'main' } });
  const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'head_turn_params' });
  const rest = device.createBuffer({ size: Math.max(16, count * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'rest_heights' });
  device.queue.writeBuffer(rest, 0, restHeights);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [params, source.positions, source.covariances, rest, target.positions, target.covariances]
      .map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const data = new ArrayBuffer(48), f = new Float32Array(data), u = new Uint32Array(data);
  if (neck) { f.set(neck.pivot, 0); f[3] = neck.baseY; f[7] = neck.topY; }
  u[8] = count;
  const write = () => device.queue.writeBuffer(params, 0, data);
  write();
  return {
    setTurn(turn) {
      if (turn && !neck) throw new Error('a head turn needs the avatar\'s neck, which this bundle lacks');
      u[9] = turn ? 1 : 0;
      if (turn) f.set([turn[0], turn[1], turn[2]], 4);
      write();
    },
    encode(encoder) {
      const pass = encoder.beginComputePass({ label: 'head_turn' });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 256));
      pass.end();
    },
    destroy() { params.destroy(); rest.destroy(); },
  };
}
