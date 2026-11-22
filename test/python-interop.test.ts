import { describe, expect, test } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fromHalf } from 'webgpu-gaussian-rasterizer';
import { avatarGaussians, parseBundle } from '../src/index.js';

const project = fileURLToPath(new URL('../python', import.meta.url));
const hasUv = spawnSync('uv', ['--version']).status === 0;

// The Python exporter writes, this package reads.
describe.skipIf(!hasUv && !import.meta.env.REQUIRE_PYTHON)('bundles written by gagavatar_web', () => {
  test('parse with the arrays and meta Python wrote', () => {
    const out = join(mkdtempSync(join(tmpdir(), 'gsb-')), 'avatar.gsb');
    const script = `
import sys, numpy as np
from gagavatar_web import write_bundle, covariances_from_scale_rotation
n, h = 9, 4
pos = np.arange(3 * n, dtype=np.float32).reshape(n, 3) / 10
write_bundle(sys.argv[1], {"count": n, "head_count": h, "tan_fov": 1 / 12, "feature_channels": 5,
                           "neck": {"pivot": [0.0, -0.4, -0.3], "base_y": -0.7, "top_y": -0.4}}, {
    "pos": pos,
    "cov": covariances_from_scale_rotation(np.full((n, 3), 0.1), np.tile([1.0, 0, 0, 0], (n, 1))),
    "opacity": np.linspace(0, 1, n, dtype=np.float32),
    "feat5": (np.arange(5 * n).reshape(n, 5) / 64).astype(np.float16),
    "head_ref": pos[:h],
})`;
    const run = spawnSync('uv', ['run', '--frozen', '--project', project, 'python', '-c', script, out], { encoding: 'utf8' });
    expect(run.status, run.stderr).toBe(0);
    const bytes = readFileSync(out);
    const g = avatarGaussians(parseBundle(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
    expect(g.count).toBe(9);
    expect(g.neck).toEqual({ pivot: [0, -0.4, -0.3], baseY: -0.7, topY: -0.4 });
    expect(g.positions[5]).toBeCloseTo(0.5, 6);
    expect([...g.covariances.subarray(0, 6)].map((v) => +v.toFixed(6))).toEqual([0.01, 0, 0, 0.01, 0, 0.01]);
    // Five channels padded to six for the rasterizer.
    expect(g.features.length).toBe(6 * 9);
    expect(fromHalf(g.features[5])).toBe(0);
    expect(fromHalf(g.features[7])).toBe(6 / 64);
  });
});
