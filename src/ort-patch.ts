/**
 * ONNX Runtime Web binds a graph-captured session's gpu-buffer inputs and
 * outputs on its first run only. The bindings keep the WebGPU handles that
 * run registered for the buffers, but every run unregisters its buffers as
 * it ends, which frees those handles. Later runs register the buffers again
 * under whatever handles the allocator gives out then. When those differ,
 * the replayed graph reads and writes the wrong buffers: the caller's
 * output never changes. Which handles come back depends on what else
 * registered one in between, so it varies with the environment, and a
 * second session on the same module fails every time.
 *
 * Later runs also miss the pre-allocated outputs, since they compare the
 * value the run returns with the tensor made for that run, which never
 * matches. The output then takes the path of one ORT allocated itself,
 * which looks its GPU buffer up through the bound handle.
 *
 * Both sites are in `run()` in js/web/lib/wasm/wasm-core-impl.ts, shipped
 * in the npm package (lines 853-948 and 1152-1167 of 1.31.1's copy). The
 * fix keeps a captured session's buffers registered after each run, and
 * treats every gpu-buffer output of a captured session as pre-allocated.
 * The names are the minifier's, so each entry applies only to
 * `dist/ort.webgpu.bundle.min.mjs` of its own version.
 */
// Once a captured session's buffers stay registered, nothing unregisters
// them, not even releasing the session. The module holds on to one
// destroyed buffer pair per captured refiner a page creates.
const PATCHES: { version: string; edits: [site: string, fixed: string][] }[] = [
  {
    version: '1.30.0',
    edits: [
      ['if(ae===O[R]||oe.includes(O[R])){', 'if(ae===O[R]||oe.includes(O[R])||A&&i[R]?.[3]==="gpu-buffer"){'],
      [
        's.forEach(K=>{K&&K[3]==="gpu-buffer"&&l.webgpuUnregisterBuffer(K[2].gpuBuffer)}),i.forEach(K=>{K&&K[3]==="gpu-buffer"&&l.webgpuUnregisterBuffer(K[2].gpuBuffer)})',
        'A||(s.forEach(K=>{K&&K[3]==="gpu-buffer"&&l.webgpuUnregisterBuffer(K[2].gpuBuffer)}),i.forEach(K=>{K&&K[3]==="gpu-buffer"&&l.webgpuUnregisterBuffer(K[2].gpuBuffer)}))',
      ],
    ],
  },
  {
    version: '1.31.1',
    edits: [
      ['if(ce===J[P]||Q.includes(J[P])){', 'if(ce===J[P]||Q.includes(J[P])||L&&i[P]?.[3]==="gpu-buffer"){'],
      [
        's.forEach(w=>{w&&w[3]==="gpu-buffer"&&d.webgpuUnregisterBuffer(w[2].gpuBuffer)}),i.forEach(w=>{w&&w[3]==="gpu-buffer"&&d.webgpuUnregisterBuffer(w[2].gpuBuffer)})',
        'L||(s.forEach(w=>{w&&w[3]==="gpu-buffer"&&d.webgpuUnregisterBuffer(w[2].gpuBuffer)}),i.forEach(w=>{w&&w[3]==="gpu-buffer"&&d.webgpuUnregisterBuffer(w[2].gpuBuffer)}))',
      ],
    ],
  },
];

const occursOnce = (source: string, site: string) => {
  const at = source.indexOf(site);
  return at >= 0 && source.indexOf(site, at + 1) < 0;
};

export const GRAPH_CAPTURE_PATCHED_VERSIONS: readonly string[] = PATCHES.map((p) => p.version);

/**
 * Applies the graph-capture fix to the source of an ONNX Runtime Web
 * `ort.webgpu.bundle.min.mjs`. Import the result (from a Blob URL, say) in
 * place of the package to use `graphCapture: true` with `createRefiner`.
 * Throws for a file it does not recognise.
 */
export function patchOrtGraphCapture(source: string): { source: string; version: string } {
  for (const { version, edits } of PATCHES) {
    if (edits.every(([, fixed]) => source.includes(fixed))) return { source, version };
    if (edits.every(([site]) => occursOnce(source, site))) {
      return { source: edits.reduce((s, [site, fixed]) => s.replace(site, () => fixed), source), version };
    }
  }
  throw new Error(`not an ort.webgpu.bundle.min.mjs of ONNX Runtime Web ${GRAPH_CAPTURE_PATCHED_VERSIONS.join(' or ')}`);
}
