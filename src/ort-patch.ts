/**
 * ONNX Runtime Web binds a graph-captured session's pre-allocated
 * gpu-buffer outputs on its first run only. Later runs recognise them by
 * comparing the value the run returns with the handle made for that run,
 * which never matches, so the output takes the path of one ORT allocated
 * itself and looks up its GPU buffer through a handle freed after the first
 * run. The second run then throws, or returns a stale buffer. The site is
 * `run()` in js/web/lib/wasm/wasm-core-impl.ts, shipped in the npm package
 * (lines 853-948 of 1.31.1's copy).
 *
 * The fix treats every gpu-buffer output of a captured session as
 * pre-allocated. The names are the minifier's, so each entry applies only to
 * `dist/ort.webgpu.bundle.min.mjs` of its own version.
 */
const PATCHES = [
  {
    version: '1.30.0',
    site: 'if(ae===O[R]||oe.includes(O[R])){',
    fixed: 'if(ae===O[R]||oe.includes(O[R])||A&&i[R]?.[3]==="gpu-buffer"){',
  },
  {
    version: '1.31.1',
    site: 'if(ce===J[P]||Q.includes(J[P])){',
    fixed: 'if(ce===J[P]||Q.includes(J[P])||L&&i[P]?.[3]==="gpu-buffer"){',
  },
] as const;

export const GRAPH_CAPTURE_PATCHED_VERSIONS: readonly string[] = PATCHES.map((p) => p.version);

/**
 * Applies the graph-capture fix to the source of an ONNX Runtime Web
 * `ort.webgpu.bundle.min.mjs`. Import the result (from a Blob URL, say) in
 * place of the package to use `graphCapture: true` with `createRefiner`.
 * Throws for a file it does not recognise.
 */
export function patchOrtGraphCapture(source: string): { source: string; version: string } {
  for (const p of PATCHES) {
    if (source.includes(p.fixed)) return { source, version: p.version };
    const at = source.indexOf(p.site);
    if (at >= 0 && source.indexOf(p.site, at + 1) < 0) {
      return { source: source.slice(0, at) + p.fixed + source.slice(at + p.site.length), version: p.version };
    }
  }
  throw new Error(`not an ort.webgpu.bundle.min.mjs of ONNX Runtime Web ${GRAPH_CAPTURE_PATCHED_VERSIONS.join(' or ')}`);
}
