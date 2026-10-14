// Counter-based random numbers: the same (seed, step, row, slot) gives the same draw on the CPU and in
// the WebGPU shaders, so a seed reproduces a line on either engine (murmur3-style integer mixing).
export function mix(x) {
  x ^= x >>> 16; x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}

export function hash4(seed, step, row, slot) {
  let h = mix((seed >>> 0) ^ 0x9e3779b9);
  h = mix(h + Math.imul(step >>> 0, 0x85ebca6b));
  h = mix(h + Math.imul(row >>> 0, 0xc2b2ae35));
  h = mix(h + Math.imul(slot >>> 0, 0x27d4eb2f));
  return h;
}

/** Uniform in [0, 1) with 24 bits, like the shader's f32(h >> 8) / 2^24. */
export function uniform(seed, step, row, slot) {
  return (hash4(seed, step, row, slot) >>> 8) / 16777216;
}

/** Two standard normals from slots (slot, slot + 1) by Box-Muller. */
export function normal2(seed, step, row, slot) {
  const u1 = 1 - uniform(seed, step, row, slot), u2 = uniform(seed, step, row, slot + 1);
  const r = Math.sqrt(-2 * Math.log(u1)), a = 2 * Math.PI * u2;
  return [r * Math.cos(a), r * Math.sin(a)];
}
