import { createPresenter, type Presenter } from 'webgpu-gaussian-rasterizer';
import {
  createAvatarRenderer, createRefiner, parseBundle, type AvatarBundle, type AvatarRenderer, type OrtModule, type Refiner, type RefinerSpec,
} from '../../src/index.js';
import { createSyntheticAvatar } from './avatar.js';

const SIZE = 512;
const status = document.querySelector<HTMLOutputElement>('#status')!;
const bodyStill = document.querySelector<HTMLInputElement>('#body-still')!;

async function main() {
  if (!('gpu' in navigator)) throw new Error('This browser has no WebGPU.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter.');
  let device = await adapter.requestDevice();
  const context = document.querySelector<HTMLCanvasElement>('#view')!.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const synthetic = createSyntheticAvatar();
  let bundle: AvatarBundle = synthetic.bundle;
  let ownVertices: Float32Array | null = null;
  let refiner: Refiner | null = null;
  let renderer: AvatarRenderer;
  let presenter: Presenter;
  // Frames wait while the renderer is rebuilt.
  let rebuilding = false;

  const rebuild = async () => {
    rebuilding = true;
    renderer?.destroy();
    const outputFormat = refiner ? refiner.spec.dataType : 'float16';
    renderer = await createAvatarRenderer(device, { bundle, size: SIZE, image: refiner?.input, outputFormat });
    presenter = createPresenter(device, {
      source: refiner ? refiner.output : renderer.image, width: SIZE, height: SIZE, sourceFormat: outputFormat, targetFormat: format,
      sharpen: refiner ? 0.5 : 0,
    });
    ownVertices = bundle === synthetic.bundle ? null : renderer.gaussians.headRef;
    rebuilding = false;
  };
  await rebuild();

  document.querySelector<HTMLInputElement>('#bundle')!.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    bundle = parseBundle(await file.arrayBuffer());
    await rebuild();
  });
  document.querySelector<HTMLInputElement>('#refiner')!.addEventListener('change', async (e) => {
    const files = [...((e.target as HTMLInputElement).files ?? [])];
    const model = files.find((f) => f.name.endsWith('.onnx')), json = files.find((f) => f.name.endsWith('.json'));
    if (!model || !json) { status.value = 'Pick the .onnx file and its .json together.'; return; }
    if (refiner) { status.value = 'Reload the page to load another refiner.'; return; }
    status.value = 'Loading the refiner...';
    const spec = JSON.parse(await json.text()) as RefinerSpec;
    if (spec.dataType === 'float16' && !adapter.features.has('shader-f16')) {
      status.value = 'This GPU has no shader-f16; export the refiner with --precision float32.';
      return;
    }
    const ort = await import('onnxruntime-web/webgpu');
    refiner = await createRefiner({ ort: ort as unknown as OrtModule, model: new Uint8Array(await model.arrayBuffer()), spec, log: (m) => { status.value = m; } });
    // From here on everything renders on ONNX Runtime's device, which the
    // refiner's buffers belong to.
    device = refiner.device;
    context.configure({ device, format, alphaMode: 'opaque' });
    await rebuild();
  });

  let frames = 0, since = performance.now();
  const frame = async (now: number) => {
    if (rebuilding) { requestAnimationFrame(frame); return; }
    const t = now / 1000;
    const yaw = 0.45 * Math.sin(0.7 * t), pitch = 0.15 * Math.sin(1.1 * t + 1);
    const camera = synthetic.camera(yaw, pitch);
    if (ownVertices) {
      // A user's bundle keeps its own camera translation and gets no mouth motion.
      for (const k of [3, 7, 11]) camera[k] = bundle.meta.camera?.[k] ?? camera[k];
    }
    renderer.setFrame({ vertices: ownVertices ?? synthetic.vertices(0.5 + 0.5 * Math.sin(6 * t)), camera, bodyStill: bodyStill.checked });
    const encoder = device.createCommandEncoder();
    renderer.encode(encoder);
    if (refiner) {
      device.queue.submit([encoder.finish()]);
      await refiner.run();
      const present = device.createCommandEncoder();
      presenter.encode(present, context.getCurrentTexture());
      device.queue.submit([present.finish()]);
    } else {
      presenter.encode(encoder, context.getCurrentTexture());
      device.queue.submit([encoder.finish()]);
    }
    frames++;
    if (now - since > 1000) {
      status.value = `${renderer.gaussians.count.toLocaleString()} Gaussians, ${renderer.gaussians.channels} channels${refiner ? ', refined' : ''}, ${((frames * 1000) / (now - since)).toFixed(0)} fps`;
      frames = 0;
      since = now;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e) => { status.value = String(e instanceof Error ? e.message : e); });
