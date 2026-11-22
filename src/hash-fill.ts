/**
 * `n` values in [-bound, bound) from a fixed integer hash of their index:
 * the input a refiner's self-check feeds the network.
 * `gagavatar_web.refiner.hash_fill` computes the same values in Python.
 */
export function hashFill(n: number, seed: number, bound: number): Float32Array {
  const out = new Float32Array(n);
  const mix = Math.imul(seed + 1, 0x9e3779b9) >>> 0;
  for (let i = 0; i < n; i++) {
    let x = (i ^ mix) >>> 0;
    x ^= x >>> 16; x = Math.imul(x, 0x7feb352d); x ^= x >>> 15; x = Math.imul(x, 0x846ca68b); x ^= x >>> 16;
    out[i] = ((x >>> 0) / 4294967296 * 2 - 1) * bound;
  }
  return out;
}
