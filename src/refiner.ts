import { fromHalf, toHalf } from 'webgpu-gaussian-rasterizer';
import { hashFill } from './hash-fill.js';

/**
 * The parts of ONNX Runtime Web (`onnxruntime-web/webgpu`, 1.30 or later)
 * the refiner uses. The caller passes the module in, so that it can load the
 * copy it wants, patched or not (see `patchOrtGraphCapture`).
 */
export interface OrtModule {
  InferenceSession: { create(model: Uint8Array, options: Record<string, unknown>): Promise<OrtSession> };
  Tensor: { fromGpuBuffer(buffer: GPUBuffer, options: { dataType: string; dims: readonly number[] }): unknown };
  env: { webgpu: { device?: GPUDevice | Promise<GPUDevice> } };
}

interface OrtSession {
  run(feeds: Record<string, unknown>, fetches: Record<string, unknown>): Promise<unknown>;
  release(): Promise<void>;
}

/** What `gagavatar_web.refiner.export_refiner` writes next to the ONNX file. */
export interface RefinerSpec {
  dataType: 'float16' | 'float32';
  input: { name: string; dims: number[] };
  output: { name: string; dims: number[] };
  /** Output values the exporter computed for a `hashFill` input, which the refiner checks at load. */
  check?: { inputSeed: number; bound: number; indices: number[]; values: number[]; tolerance: number };
}

export interface RefinerOptions {
  ort: OrtModule;
  model: Uint8Array;
  /** The model's external data files, by the path the graph names them with. */
  externalData?: { path: string; data: Uint8Array }[];
  spec: RefinerSpec;
  /**
   * Captures the session's GPU work once and replays it. ONNX Runtime Web
   * up to at least 1.31.1 needs `patchOrtGraphCapture` for this. An
   * unpatched copy fails or works depending on whether its allocator hands
   * a freed address back. The refiner probes the session at load and runs
   * without capture when the probe fails, but a probe that passes does not
   * rule out a failure later.
   */
  graphCapture?: boolean;
  log?: (message: string) => void;
}

export interface RefinerCheck {
  ok: boolean;
  maxDiff: number;
  tolerance: number;
}

export interface Refiner {
  /** ONNX Runtime's WebGPU device; render the input on this one. */
  readonly device: GPUDevice;
  /** The network's input, which `createAvatarRenderer({ image })` renders into. */
  readonly input: GPUBuffer;
  /** The network's output, planar, in `spec.dataType`. */
  readonly output: GPUBuffer;
  readonly spec: RefinerSpec;
  readonly graphCapture: boolean;
  /** null when the spec carries no check. */
  readonly check: RefinerCheck | null;
  /** Runs the network on whatever the input buffer holds when the GPU gets to it. */
  run(): Promise<void>;
  release(): Promise<void>;
}

const elements = (dims: readonly number[]) => dims.reduce((a, b) => a * b, 1);

/**
 * Loads the network that turns the splatted feature image into the
 * picture (GAGAvatar's upsampler, or a network trained to stand in for
 * it), on ONNX Runtime Web's WebGPU backend, with its input and output in
 * GPU buffers of its own.
 */
export async function createRefiner(options: RefinerOptions): Promise<Refiner> {
  const { ort, model, externalData, spec, log = () => {} } = options;
  const fp16 = spec.dataType === 'float16';
  const bytesPerElement = fp16 ? 2 : 4;
  const nIn = elements(spec.input.dims), nOut = elements(spec.output.dims);

  const open = async (capture: boolean) => {
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      preferredOutputLocation: 'gpu-buffer',
      ...(externalData ? { externalData } : {}),
      ...(capture ? { enableGraphCapture: true } : {}),
    });
    const device = await ort.env.webgpu.device;
    if (!device) throw new Error('ONNX Runtime Web has no WebGPU device after creating the session');
    const U = GPUBufferUsage;
    const input = device.createBuffer({ size: nIn * bytesPerElement, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST, label: 'refiner_input' });
    const output = device.createBuffer({ size: nOut * bytesPerElement, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST, label: 'refiner_output' });
    const feeds = { [spec.input.name]: ort.Tensor.fromGpuBuffer(input, { dataType: spec.dataType, dims: spec.input.dims }) };
    const fetches = { [spec.output.name]: ort.Tensor.fromGpuBuffer(output, { dataType: spec.dataType, dims: spec.output.dims }) };
    return { session, device, input, output, capture, run: async () => { await session.run(feeds, fetches); } };
  };

  const write = (device: GPUDevice, buffer: GPUBuffer, values: Float32Array) =>
    device.queue.writeBuffer(buffer, 0, fp16 ? Uint16Array.from(values, toHalf) : values);
  const read = async (device: GPUDevice, buffer: GPUBuffer): Promise<Float32Array> => {
    const staging = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = staging.getMappedRange().slice(0);
    staging.destroy();
    return fp16 ? Float32Array.from(new Uint16Array(raw), fromHalf) : new Float32Array(raw);
  };
  const probeInput = hashFill(nIn, spec.check?.inputSeed ?? 0, spec.check?.bound ?? 1);
  const compare = (out: Float32Array): RefinerCheck | null => {
    if (!spec.check) return null;
    const { indices, values, tolerance } = spec.check;
    const maxDiff = Math.max(...indices.map((idx, k) => Math.abs(out[idx] - values[k])));
    return { ok: out.every(Number.isFinite) && maxDiff < tolerance, maxDiff, tolerance };
  };

  // A captured session replays the first run's work, so its faults show on
  // later runs: on an unpatched ONNX Runtime Web the second run can throw,
  // or hand back the first run's output. Three runs on two inputs catch both.
  const probe = async (s: Awaited<ReturnType<typeof open>>) => {
    write(s.device, s.input, probeInput);
    await s.run();
    const first = await read(s.device, s.output);
    if (!s.capture) return compare(first);
    write(s.device, s.input, probeInput.map((v) => -v));
    await s.run();
    const other = await read(s.device, s.output);
    write(s.device, s.input, probeInput);
    await s.run();
    const again = await read(s.device, s.output);
    if (other.every((v, i) => v === first[i])) throw new Error('a new input left the output unchanged');
    if (again.some((v, i) => v !== first[i])) throw new Error('the same input gave a different output');
    return compare(again);
  };

  let s = await open(options.graphCapture ?? false);
  let check: RefinerCheck | null;
  try {
    check = await probe(s);
  } catch (e) {
    if (!s.capture) throw e;
    log(`graph capture failed (${(e as Error).message}); running without it`);
    await s.session.release();
    s.input.destroy(); s.output.destroy();
    s = await open(false);
    check = await probe(s);
  }
  if (check && !check.ok) log(`refiner check failed: largest difference ${check.maxDiff} against a tolerance of ${check.tolerance}`);
  const { session, device, input, output, capture, run } = s;
  return {
    device, input, output, spec, graphCapture: capture, check, run,
    async release() {
      await session.release();
      input.destroy(); output.destroy();
    },
  };
}
