import { describe, expect, test } from 'vitest';
import { createGaussianRasterizer } from 'webgpu-gaussian-rasterizer';
import {
  avatarGaussians, createAvatarRenderer, neckWeight, rasterizerCamera, rotationMatrix, takeHeadTurnOffCamera,
} from '../src/index.js';
import { createSyntheticAvatar } from '../examples/synthetic/avatar.js';
import { psnr, readBuffer, useDevice } from './gpu.js';

const device = useDevice();
const SIZE = 128;

// The same frame drawn by the rasterizer alone, from Gaussians posed (and,
// for a still body, turned) on the CPU.
async function reference(avatar: ReturnType<typeof createSyntheticAvatar>, vertices: Float32Array, camera: Float32Array, bodyStill: boolean) {
  const g = avatarGaussians(avatar.bundle);
  const positions = g.positions.slice();
  positions.set(vertices);
  const covariances = g.covariances.slice();
  let cam = camera;
  if (bodyStill) {
    cam = new Float32Array(12);
    const turn = takeHeadTurnOffCamera(camera, cam);
    for (let i = 0; i < g.count; i++) {
      const w = neckWeight(g.neck!, g.positions[3 * i + 1]);
      const m = rotationMatrix(turn.map((t) => t * w));
      const p = [0, 1, 2].map((k) => positions[3 * i + k] - g.neck!.pivot[k]);
      for (let r = 0; r < 3; r++) positions[3 * i + r] = m[3 * r] * p[0] + m[3 * r + 1] * p[1] + m[3 * r + 2] * p[2] + g.neck!.pivot[r];
      const c = covariances.subarray(6 * i, 6 * i + 6);
      const s = [c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5]];
      const ms = [0, 1, 2].flatMap((r) => [0, 1, 2].map((k) => m[3 * r] * s[k] + m[3 * r + 1] * s[3 + k] + m[3 * r + 2] * s[6 + k]));
      const t = (r: number, k: number) => ms[3 * r] * m[3 * k] + ms[3 * r + 1] * m[3 * k + 1] + ms[3 * r + 2] * m[3 * k + 2];
      c.set([t(0, 0), t(0, 1), t(0, 2), t(1, 1), t(1, 2), t(2, 2)]);
    }
  }
  const r = await createGaussianRasterizer(device(), { capacity: g.count, width: SIZE, height: SIZE, channels: g.channels, outputFormat: 'float32' });
  r.writePositions(positions);
  r.writeCovariances(covariances);
  r.writeOpacities(g.opacities);
  r.writeFeatures(g.features);
  r.setCamera(rasterizerCamera(cam, SIZE, g.tanFov));
  r.render();
  const image = new Float32Array(await readBuffer(device(), r.buffers.image));
  r.destroy();
  return image;
}

async function render(avatar: ReturnType<typeof createSyntheticAvatar>, vertices: Float32Array, camera: Float32Array, bodyStill: boolean) {
  const renderer = await createAvatarRenderer(device(), { bundle: avatar.bundle, size: SIZE, outputFormat: 'float32' });
  renderer.setFrame({ vertices, camera, bodyStill });
  const encoder = device().createCommandEncoder();
  renderer.encode(encoder);
  device().queue.submit([encoder.finish()]);
  const image = new Float32Array(await readBuffer(device(), renderer.image));
  renderer.destroy();
  return image;
}

describe('createAvatarRenderer', () => {
  const avatar = createSyntheticAvatar({ rings: 12, segments: 16, bodyCount: 800 });

  test('draws the head at the frame\'s vertices, as the rasterizer alone does', async () => {
    const vertices = avatar.vertices(0.8), camera = avatar.camera(0.3, 0.1);
    const image = await render(avatar, vertices, camera, false);
    expect(image).toEqual(await reference(avatar, vertices, camera, false));
    // The head covers the image's centre.
    const centre = SIZE * (SIZE / 2) + SIZE / 2;
    expect(image[centre]).toBeGreaterThan(0.5);
    // The mouth moved the image.
    expect(psnr(image, await render(avatar, avatar.vertices(0), camera, false))).toBeLessThan(60);
  });

  test('turns the head on the neck and keeps the body still', async () => {
    const vertices = avatar.vertices(0.3), camera = avatar.camera(0.35, -0.15);
    const image = await render(avatar, vertices, camera, true);
    expect(psnr(image, await reference(avatar, vertices, camera, true))).toBeGreaterThan(60);
    // The body is where the zero-pose camera sees it, not where the turned camera does.
    const still = await render(avatar, vertices, avatar.camera(0, 0), false);
    const turnedBust = await render(avatar, vertices, camera, false);
    const bottom = (img: Float32Array) => img.filter((_, k) => Math.floor((k % (SIZE * SIZE)) / SIZE) > SIZE * 0.85);
    expect(psnr(bottom(image), bottom(still))).toBeGreaterThan(psnr(bottom(image), bottom(turnedBust)) + 10);
  });

  test('refuses a head turn for a bundle without a neck', async () => {
    const { neck: _, ...meta } = avatar.bundle.meta;
    const renderer = await createAvatarRenderer(device(), { bundle: { meta, arrays: avatar.bundle.arrays }, size: SIZE });
    expect(() => renderer.setFrame({ camera: avatar.camera(0.2, 0), bodyStill: true })).toThrow(/neck/);
    renderer.destroy();
  });
});
