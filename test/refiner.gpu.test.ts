import { describe, expect, test } from 'vitest';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRefiner, hashFill, patchOrtGraphCapture, type OrtModule, type RefinerSpec } from '../src/index.js';
import { SCALE_MODEL_BASE64 } from './fixtures/scale-model.js';
import { gpu, readBuffer, useDevice } from './gpu.js';

// ONNX Runtime Web finds its adapter through navigator.gpu, which Dawn
// stands in for here.
Object.defineProperty(globalThis.navigator, 'gpu', { value: gpu, configurable: true });
// Only for the REQUIRE_WEBGPU check and skipping; the refiner uses ONNX Runtime's own device.
useDevice();

const require = createRequire(import.meta.url);
const distDir = dirname(require.resolve('onnxruntime-web/webgpu'));

async function importPatchedOrt(): Promise<OrtModule & { env: { wasm: { numThreads: number; wasmPaths: string } } }> {
  const { source } = patchOrtGraphCapture(readFileSync(join(distDir, 'ort.webgpu.bundle.min.mjs'), 'utf8'));
  const file = join(mkdtempSync(join(tmpdir(), 'ort-')), 'ort.webgpu.bundle.min.mjs');
  // Without its source map next to it, which Vite would warn about.
  writeFileSync(file, source.replace(/\/\/# sourceMappingURL=\S+\s*$/, ''));
  const ort = await import(/* @vite-ignore */ pathToFileURL(file).href);
  // The wasm files stay in the package, not next to the copy.
  ort.env.wasm.wasmPaths = pathToFileURL(distDir + '/').href;
  ort.env.wasm.numThreads = 1;
  return ort;
}

const model = Uint8Array.from(atob(SCALE_MODEL_BASE64), (c) => c.charCodeAt(0));
const seed = 3;
const spec: RefinerSpec = {
  dataType: 'float32',
  input: { name: 'x', dims: [1, 4] },
  output: { name: 'y', dims: [1, 4] },
  check: { inputSeed: seed, bound: 1, indices: [0, 1, 2, 3], values: [...hashFill(4, seed, 1)].map((v) => 2 * v + 1), tolerance: 1e-5 },
};

describe('createRefiner', () => {
  test('runs the network on its input buffer, with graph capture on a patched ONNX Runtime Web', async () => {
    const ort = await importPatchedOrt();
    const logs: string[] = [];
    const refiner = await createRefiner({ ort, model, spec, graphCapture: true, log: (m) => logs.push(m) });
    expect(logs).toEqual([]);
    expect(refiner.graphCapture).toBe(true);
    expect(refiner.check?.ok).toBe(true);
    for (let k = 0; k < 3; k++) {
      refiner.device.queue.writeBuffer(refiner.input, 0, new Float32Array([k, k + 1, k + 2, k + 3]));
      await refiner.run();
      expect([...new Float32Array(await readBuffer(refiner.device, refiner.output))]).toEqual([0, 1, 2, 3].map((i) => 2 * (k + i) + 1));
    }
    // A check the network cannot pass is reported, not thrown.
    const wrong = await createRefiner({ ort, model, spec: { ...spec, check: { ...spec.check!, values: [9, 9, 9, 9] } }, log: (m) => logs.push(m) });
    expect(wrong.check?.ok).toBe(false);
    expect(logs).toEqual([expect.stringMatching(/check failed/)]);
    await wrong.release();
    await refiner.release();
  });
});
