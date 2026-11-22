import { afterAll, afterEach, beforeAll, beforeEach, expect } from 'vitest';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);

// The GPU object has to outlive the device, or Dawn crashes when it is collected.
export const gpu = create([]);

/**
 * Registers hooks that open a Dawn device before the file's tests, skip
 * them without an adapter (or fail them, under REQUIRE_WEBGPU, so that CI
 * cannot pass by skipping), and fail any test that raises a WebGPU error.
 */
export function useDevice(): () => GPUDevice {
  let device: GPUDevice | undefined;
  const errors: string[] = [];
  beforeAll(async () => {
    device = await (await gpu.requestAdapter())?.requestDevice();
    device?.addEventListener('uncapturederror', (e) => errors.push((e as GPUUncapturedErrorEvent).error.message));
  });
  beforeEach(({ skip }) => {
    if (device) return;
    if (import.meta.env.REQUIRE_WEBGPU) throw new Error('REQUIRE_WEBGPU is set, but Dawn found no WebGPU adapter');
    skip();
  });
  afterEach(() => expect(errors.splice(0)).toEqual([]));
  afterAll(() => device?.destroy());
  return () => device!;
}

export async function readBuffer(device: GPUDevice, buffer: GPUBuffer, bytes = buffer.size): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.destroy();
  return out;
}

export function psnr(a: Float32Array, b: Float32Array): number {
  let se = 0;
  for (let i = 0; i < a.length; i++) se += (a[i] - b[i]) ** 2;
  return se === 0 ? Infinity : 10 * Math.log10(a.length / se);
}
