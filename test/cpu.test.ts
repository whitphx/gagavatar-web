import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import {
  avatarGaussians, encodeBundle, hashFill, parseBundle, patchOrtGraphCapture, rotationMatrix,
  takeHeadTurnOffCamera, viewFromCamera, createPacer, type PacedFrame,
} from '../src/index.js';
import { createSyntheticAvatar } from '../examples/synthetic/avatar.js';

describe('bundles', () => {
  test('round-trip through encodeBundle and parseBundle', () => {
    const { bundle } = createSyntheticAvatar({ bodyCount: 50 });
    const bytes = encodeBundle(bundle.meta, bundle.arrays);
    expect(new TextDecoder().decode(new Uint8Array(bytes, 0, 4))).toBe('GSB1');
    const back = parseBundle(bytes);
    expect(back.meta).toEqual(bundle.meta);
    for (const [name, a] of Object.entries(bundle.arrays)) {
      expect(back.arrays[name].shape).toEqual(a.shape);
      expect(back.arrays[name].data).toEqual(a.data);
      // Viewed in place: aligned, so no copy.
      expect(back.arrays[name].data.buffer).toBe(bytes);
    }
  });

  test('pads odd channel counts to whole words for the rasterizer', () => {
    const { bundle } = createSyntheticAvatar({ bodyCount: 10, channels: 3 });
    const g = avatarGaussians(bundle);
    const raw = bundle.arrays.feat3.data as Uint16Array;
    expect(g.features.length).toBe(4 * g.count);
    expect([...g.features.subarray(4, 7)]).toEqual([...raw.subarray(3, 6)]);
    expect(g.features[7]).toBe(0);
  });

  test('rejects arrays whose shape disagrees with the header', () => {
    const { bundle } = createSyntheticAvatar({ bodyCount: 10 });
    const meta = { ...bundle.meta, count: bundle.meta.count + 1 };
    expect(() => avatarGaussians({ meta, arrays: bundle.arrays })).toThrow(/expected/);
    expect(() => parseBundle(new TextEncoder().encode('GSB2xxxxxxxxxxxx').buffer)).toThrow(/not a GSB1/);
  });
});

describe('cameras and the neck turn', () => {
  test('viewFromCamera moves GAGAvatar\'s view axes to OpenCV\'s', () => {
    const { camera } = createSyntheticAvatar({ bodyCount: 0 });
    const v = viewFromCamera(camera(0, 0));
    // A point above the head's centre lands above the image centre (negative y)
    // and in front of the camera.
    const p = [0, 1, 0, 1];
    const t = [0, 1, 2].map((r) => v[r] * p[0] + v[4 + r] * p[1] + v[8 + r] * p[2] + v[12 + r] * p[3]);
    expect(t[1]).toBeLessThan(0);
    expect(t[2]).toBeCloseTo(5000 / 512, 5);
  });

  test('takeHeadTurnOffCamera returns the turn that makes the zero-pose camera see the same head', () => {
    const { camera } = createSyntheticAvatar({ bodyCount: 0 });
    const posed = camera(0.4, -0.2);
    const still = new Float32Array(12);
    const turn = takeHeadTurnOffCamera(posed, still);
    // Adding 0 turns -0 into 0, which toEqual tells apart.
    expect([...still].map((v) => v + 0)).toEqual([...camera(0, 0)].map((v) => v + 0));
    const m = rotationMatrix(turn);
    // x_view = x R + T, with R the 3x4 matrix's upper 3x3 and T its last column.
    const rowView = (cam: Float32Array, x: number[]) => [0, 1, 2].map((j) => x[0] * cam[j] + x[1] * cam[4 + j] + x[2] * cam[8 + j] + cam[j * 4 + 3]);
    const x = [0.3, -0.2, 0.4];
    const turned = [0, 1, 2].map((r) => m[3 * r] * x[0] + m[3 * r + 1] * x[1] + m[3 * r + 2] * x[2]);
    const a = rowView(posed, x), b = rowView(still, turned);
    for (let k = 0; k < 3; k++) expect(b[k]).toBeCloseTo(a[k], 5);
  });
});

describe('hashFill', () => {
  // gagavatar_web's Python tests pin the same values.
  test('gives the values the Python exporter checks against', () => {
    const v = hashFill(8, 3, 1);
    expect([...v.subarray(0, 4)].map((x) => x.toFixed(6))).toEqual(HASH_FILL_SEED3);
    expect(Math.max(...hashFill(10000, 0, 2).map(Math.abs))).toBeLessThanOrEqual(2);
  });
});
const HASH_FILL_SEED3 = ['-0.159008', '-0.320069', '-0.061774', '-0.906509'];

describe('patchOrtGraphCapture', () => {
  const require = createRequire(import.meta.url);
  const bundlePath = require.resolve('onnxruntime-web/webgpu').replace(/ort\.webgpu\.[^/]*$/, 'ort.webgpu.bundle.min.mjs');
  const source = readFileSync(bundlePath, 'utf8');

  test('patches the installed ONNX Runtime Web once and recognises its own output', () => {
    const { source: patched, version } = patchOrtGraphCapture(source);
    expect(version).toBe('1.31.1');
    expect(patched.length).toBe(source.length + '||L&&i[P]?.[3]==="gpu-buffer"'.length + 'L||()'.length);
    expect(patchOrtGraphCapture(patched).source).toBe(patched);
  });

  test('refuses a file it does not know', () => {
    expect(() => patchOrtGraphCapture('export {}')).toThrow(/not an ort\.webgpu\.bundle\.min\.mjs/);
  });
});

describe('createPacer', () => {
  const frame = (mediaMs: number, generation = 0): PacedFrame => ({ mediaMs, generation });
  const due = (f: PacedFrame) => f.mediaMs;

  test('renders ahead and presents each frame when it is due', () => {
    const pacer = createPacer({ slots: 4, periodMs: 40 });
    for (let k = 0; k < 6; k++) pacer.add(frame(40 * k));
    const jobs = [];
    for (let job; (job = pacer.takeNext(0, due, 0)); ) jobs.push(job);
    // Four slots, so four frames render ahead.
    expect(jobs.map((j) => j.frame.mediaMs)).toEqual([0, 40, 80, 120]);
    for (const job of jobs) pacer.finish(job, 0, due);
    expect(pacer.present(0, due)?.frame.mediaMs).toBe(0);
    expect(pacer.present(20, due)).toBeNull();
    expect(pacer.present(45, due)?.frame.mediaMs).toBe(40);
    // Presenting freed the first slot, so the next frame can render.
    expect(pacer.takeNext(45, due, 0)?.frame.mediaMs).toBe(160);
    expect(pacer.counts.presented).toBe(2);
  });

  test('skips frames that come too late or belong to another generation', () => {
    const pacer = createPacer({ slots: 4, periodMs: 40 });
    pacer.add(frame(0, 1));
    pacer.add(frame(40));
    pacer.add(frame(80));
    const job = pacer.takeNext(200, due, 0);
    // 40 is overdue with 80 queued behind it.
    expect(job?.frame.mediaMs).toBe(80);
    expect(pacer.counts.skippedOtherGeneration).toBe(1);
    expect(pacer.counts.skippedLate).toBe(1);
    pacer.finish(job!, 200, due);
    pacer.present(200, due);
    pacer.add(frame(60));
    expect(pacer.counts.skippedLate).toBe(2);
  });

  test('keeps a full ring and lets a late arrival take the last slot', () => {
    const pacer = createPacer({ slots: 2, periodMs: 40 });
    for (const t of [0, 80, 120]) pacer.add(frame(t));
    for (let job; (job = pacer.takeNext(0, due, 0)); ) pacer.finish(job, 0, due);
    pacer.add(frame(40));
    const job = pacer.takeNext(0, due, 0);
    expect(job?.frame.mediaMs).toBe(40);
    expect(pacer.counts.requeuedRingFull).toBe(1);
  });
});

