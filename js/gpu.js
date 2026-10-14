// The generator and the readers on the GPU through WebGPU: the same math as synth_cpu.js / reader_cpu.js,
// batched over the candidates. A line is about 1300 sequential steps, so the cost is per step and 64
// candidates cost about what one does. Steps are recorded in blocks into one command buffer; between
// blocks the host reads back only the done flags, and the sampled points once at the end.
import { encodeText, N_CHARS } from './vocab.js';
import { readerFeats } from './reader_cpu.js';

const TILE_ROWS = 16, TILE_K = 32, COLS = 64, WG = 64, MAXWG = 65535, SLOT = 256;
const padK = k => Math.ceil(k / TILE_K) * TILE_K;
const align = n => Math.ceil(n / 64) * 64;          // float offsets at 256-byte boundaries

const COMMON = /* wgsl */`
fn gidx(g: vec3u) -> u32 { return g.x + g.y * ${MAXWG * WG}u; }
fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
fn tanh_s(x: f32) -> f32 { let e = exp(2.0 * clamp(x, -15.0, 15.0)); return (e - 1.0) / (e + 1.0); }
fn mix(v: u32) -> u32 { var x = v; x ^= x >> 16u; x *= 0x85ebca6bu; x ^= x >> 13u; x *= 0xc2b2ae35u; x ^= x >> 16u; return x; }
fn hash4(seed: u32, step: u32, row: u32, slot: u32) -> u32 {
  var h = mix(seed ^ 0x9e3779b9u);
  h = mix(h + step * 0x85ebca6bu);
  h = mix(h + row * 0xc2b2ae35u);
  h = mix(h + slot * 0x27d4eb2fu);
  return h;
}
fn uniform01(seed: u32, step: u32, row: u32, slot: u32) -> f32 { return f32(hash4(seed, step, row, slot) >> 8u) / 16777216.0; }
`;

// Y[r, j] = B[j] + sum_k X[r, k] * W[k, j]: 64 columns per workgroup, 16 rows per thread, K in tiles of 32.
// X has row stride K (any K); W has at least padK(K) rows, the extra ones zero.
const GEMM = /* wgsl */`
struct P { rows: u32, K: u32, N: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> xs: array<f32, ${TILE_ROWS * TILE_K}>;
@compute @workgroup_size(${COLS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let j = wg.x * ${COLS}u + li.x;
  let r0 = wg.y * ${TILE_ROWS}u;
  var acc: array<f32, ${TILE_ROWS}>;
  for (var i = 0u; i < ${TILE_ROWS}u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < p.K; k0 += ${TILE_K}u) {
    for (var e = li.x; e < ${TILE_ROWS * TILE_K}u; e += ${COLS}u) {
      let r = r0 + e / ${TILE_K}u;
      let k = k0 + e % ${TILE_K}u;
      xs[e] = select(0.0, X[r * p.K + k], r < p.rows && k < p.K);
    }
    workgroupBarrier();
    if (j < p.N) {
      for (var kk = 0u; kk < ${TILE_K}u; kk++) {
        let w = W[(k0 + kk) * p.N + j];
        for (var rr = 0u; rr < ${TILE_ROWS}u; rr++) { acc[rr] += w * xs[rr * ${TILE_K}u + kk]; }
      }
    }
    workgroupBarrier();
  }
  if (j < p.N) {
    for (var rr = 0u; rr < ${TILE_ROWS}u; rr++) {
      let r = r0 + rr;
      if (r < p.rows) { Y[r * p.N + j] = acc[rr] + B[j]; }
    }
  }
}`;

// ------------------------------------------------------------------ generator kernels
// All recurrent state lives in one float buffer at the offsets the uniform names (o*).
const SYNTH = /* wgsl */`
${COMMON}
struct P { n: u32, U: u32, K: u32, M: u32, H: u32, S: u32, C: u32, K1: u32, K2: u32, Kh: u32, seed: u32, mode: u32, layer: u32, steps_out: u32,
           oX: u32, oW: u32, oH1: u32, oC1: u32, oH2: u32, oC2: u32, oH3: u32, oC3: u32, oIN1: u32, oIN2: u32, oG: u32, oABK: u32,
           oKAPPA: u32, oPHI: u32, oY: u32, oHIN: u32, oDONE: u32, oIDS: u32, bias: f32, end_frac: f32, sx: f32, sy: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> misc: array<u32>;   // [0] = step, then done (n) at oDONE, ids (U) at oIDS
@group(0) @binding(2) var<storage, read_write> out: array<f32>;    // steps_out x n x 3
@group(0) @binding(3) var<storage, read> prime: array<f32>;        // rows to feed in mode 1
@group(0) @binding(4) var<storage, read> style: array<f32>;        // S
@group(0) @binding(5) var<storage, read_write> st: array<f32>;     // the recurrent state
@group(0) @binding(6) var<storage, read_write> gin: array<f32>;    // gemm inputs: in1, in2, head input
@group(0) @binding(7) var<storage, read_write> gout: array<f32>;   // gemm outputs: gates, window, head

// in1[r] = [x, style, w, h1]
@compute @workgroup_size(${WG}) fn gather1(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.K1) { return; }
  let r = idx / p.K1; let k = idx % p.K1;
  var v = 0.0;
  if (k < 3u) { v = st[p.oX + r * 3u + k]; }
  else if (k < 3u + p.S) { v = style[k - 3u]; }
  else if (k < 3u + p.S + p.C) { v = st[p.oW + r * p.C + k - 3u - p.S]; }
  else if (k < 3u + p.S + p.C + p.H) { v = st[p.oH1 + r * p.H + k - 3u - p.S - p.C]; }
  gin[p.oIN1 + idx] = v;
}
// in2[r] = [x, style, h(layer below), w, h(own layer)]
fn gather_mid(idx: u32, oBelow: u32, oOwn: u32) {
  let r = idx / p.K2; let k = idx % p.K2;
  var v = 0.0;
  if (k < 3u) { v = st[p.oX + r * 3u + k]; }
  else if (k < 3u + p.S) { v = style[k - 3u]; }
  else if (k < 3u + p.S + p.H) { v = st[oBelow + r * p.H + k - 3u - p.S]; }
  else if (k < 3u + p.S + p.H + p.C) { v = st[p.oW + r * p.C + k - 3u - p.S - p.H]; }
  else if (k < 3u + p.S + p.H + p.C + p.H) { v = st[oOwn + r * p.H + k - 3u - p.S - p.H - p.C]; }
  gin[p.oIN2 + idx] = v;
}
@compute @workgroup_size(${WG}) fn gather2(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.K2) { return; }
  gather_mid(idx, p.oH1, p.oH2);
}
@compute @workgroup_size(${WG}) fn gather3(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.K2) { return; }
  gather_mid(idx, p.oH2, p.oH3);
}
// the LSTM cell of layer p.layer from the gates in G
@compute @workgroup_size(${WG}) fn cell(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.H) { return; }
  let r = idx / p.H; let u = idx % p.H; let base = p.oG + r * 4u * p.H;
  var oh = p.oH1; var oc = p.oC1;
  if (p.layer == 2u) { oh = p.oH2; oc = p.oC2; } else if (p.layer == 3u) { oh = p.oH3; oc = p.oC3; }
  let i = sigm(gout[base + u]); let f = sigm(gout[base + p.H + u]); let gg = tanh_s(gout[base + 2u * p.H + u]); let o = sigm(gout[base + 3u * p.H + u]);
  let cn = f * st[oc + idx] + i * gg;
  st[oc + idx] = cn;
  st[oh + idx] = o * tanh_s(cn);
}
// the window: kappa moves forward, phi over the characters (plus one past the end), w = the mix of one-hots
@compute @workgroup_size(${WG}) fn window(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.K) { return; }
  let r = idx / p.K; let j = idx % p.K;
  st[p.oKAPPA + idx] += exp(gout[p.oABK + r * 3u * p.K + 2u * p.K + j]) / 25.0;
}
@compute @workgroup_size(${WG}) fn attend(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * (p.U + 1u)) { return; }
  let r = idx / (p.U + 1u); let u = f32(idx % (p.U + 1u));
  var s = 0.0;
  for (var j = 0u; j < p.K; j++) {
    let d = st[p.oKAPPA + r * p.K + j] - u;
    s += exp(gout[p.oABK + r * 3u * p.K + j]) * exp(-exp(gout[p.oABK + r * 3u * p.K + p.K + j]) * d * d);
  }
  st[p.oPHI + idx] = s;
}
@compute @workgroup_size(${WG}) fn mixw(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.C) { return; }
  let r = idx / p.C; let c = idx % p.C;
  var s = 0.0;
  if (c != 0u) { for (var u = 0u; u < p.U; u++) { if (misc[p.oIDS + u] == c) { s += st[p.oPHI + r * (p.U + 1u) + u]; } } }
  st[p.oW + idx] = s;
}
@compute @workgroup_size(${WG}) fn headin(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.Kh) { return; }
  let r = idx / p.Kh; let k = idx % p.Kh;
  var v = 0.0;
  if (k < p.H) { v = st[p.oH1 + r * p.H + k]; } else if (k < 2u * p.H) { v = st[p.oH2 + r * p.H + k - p.H]; } else if (k < 3u * p.H) { v = st[p.oH3 + r * p.H + k - 2u * p.H]; }
  gin[p.oHIN + idx] = v;
}
// draw the next point from the mixture (mode 0) or feed the next priming row (mode 1)
@compute @workgroup_size(${WG}) fn sample(@builtin(global_invocation_id) g: vec3u) {
  let r = gidx(g); if (r >= p.n) { return; }
  let step = misc[0];
  if (p.mode == 1u) {
    let nb = (step + 1u) * 3u;
    st[p.oX + r * 3u] = prime[nb]; st[p.oX + r * 3u + 1u] = prime[nb + 1u]; st[p.oX + r * 3u + 2u] = prime[nb + 2u];
    return;
  }
  let M = p.M; let yb = p.oY + r * (1u + 6u * M); let t = 1.0 + p.bias;
  var mx = -1e30;
  for (var k = 0u; k < M; k++) { mx = max(mx, gout[yb + 1u + k] * t); }
  var z = 0.0;
  for (var k = 0u; k < M; k++) { z += exp(gout[yb + 1u + k] * t - mx); }
  let u = uniform01(p.seed, step, r, 0u);
  var k = 0u; var acc = 0.0;
  for (; k < M - 1u; k++) { acc += exp(gout[yb + 1u + k] * t - mx) / z; if (u < acc) { break; } }
  let u1 = 1.0 - uniform01(p.seed, step, r, 1u); let u2 = uniform01(p.seed, step, r, 2u);
  let rad = sqrt(-2.0 * log(u1)); let ang = 6.283185307179586 * u2;
  let z1 = rad * cos(ang); let z2 = rad * sin(ang);
  let sx = exp(gout[yb + 1u + 3u * M + k] - p.bias); let sy = exp(gout[yb + 1u + 4u * M + k] - p.bias); let rho = tanh_s(gout[yb + 1u + 5u * M + k]);
  let dx = gout[yb + 1u + M + k] + sx * z1;
  let dy = gout[yb + 1u + 2u * M + k] + sy * (rho * z1 + sqrt(max(0.0, 1.0 - rho * rho)) * z2);
  let pen = select(0.0, 1.0, uniform01(p.seed, step, r, 3u) < sigm(gout[yb]));
  var peak = 0.0;
  for (var uu = 0u; uu < p.U; uu++) { peak = max(peak, st[p.oPHI + r * (p.U + 1u) + uu]); }
  if (misc[p.oDONE + r] == 0u && step < p.steps_out) {
    let ob = (step * p.n + r) * 3u;
    out[ob] = dx * p.sx; out[ob + 1u] = dy * p.sy; out[ob + 2u] = pen;
    if (st[p.oPHI + r * (p.U + 1u) + p.U] > p.end_frac * peak && pen > 0.0) { misc[p.oDONE + r] = step + 1u; }
  }
  st[p.oX + r * 3u] = dx; st[p.oX + r * 3u + 1u] = dy; st[p.oX + r * 3u + 2u] = pen;
}
@compute @workgroup_size(1) fn tick() { misc[0] += 1u; }
`;

// ------------------------------------------------------------------ reader kernels
const READER = /* wgsl */`
${COMMON}
struct P { n: u32, Tmax: u32, Tin: u32, Cin: u32, Cout: u32, stride: u32, H: u32, In: u32, Kp: u32, step: u32, C: u32,
           olen: u32, olen_in: u32, oHS: u32, oCS: u32, pad1: u32, pad2: u32, pad3: u32, pad4: u32, pad5: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> lens: array<u32>;         // valid frames per row, per stage (offsets olen / olen_in)
@group(0) @binding(2) var<storage, read> xin: array<f32>;          // n x Tin x Cin  (or n x Tmax x In)
@group(0) @binding(3) var<storage, read> wgt: array<f32>;          // conv: Cout x Cin x 5 ; project: C x In
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> xout: array<f32>;   // conv/project output; for a layer: its output n x Tmax x 2H
@group(0) @binding(6) var<storage, read_write> rs: array<f32>;     // h and c per (direction, row)
@group(0) @binding(7) var<storage, read_write> gin: array<f32>;    // gemm input (2n x Kp, direction-major)
@group(0) @binding(8) var<storage, read_write> gates: array<f32>;  // gemm output (2n x 4H)

// conv1d, kernel 5, padding 2, ReLU; positions past the row's own length count as zero
@compute @workgroup_size(${WG}) fn conv(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.Tmax * p.Cout) { return; }
  let o = idx % p.Cout; let t = (idx / p.Cout) % p.Tmax; let r = idx / (p.Cout * p.Tmax);
  if (t >= lens[p.olen + r]) { xout[idx] = 0.0; return; }
  var acc = bias[o];
  for (var j = 0u; j < 5u; j++) {
    let ti = i32(t * p.stride + j) - 2;
    if (ti < 0 || u32(ti) >= lens[p.olen_in + r]) { continue; }
    let xb = (r * p.Tin + u32(ti)) * p.Cin;
    for (var i = 0u; i < p.Cin; i++) { acc += wgt[(o * p.Cin + i) * 5u + j] * xin[xb + i]; }
  }
  xout[idx] = max(acc, 0.0);
}
// frame of row r at this step for direction d: forward s, backward len-1-s; -1 when the row is done
fn frame(r: u32, d: u32) -> i32 {
  let L = lens[p.olen + r];
  if (p.step >= L) { return -1; }
  return select(i32(p.step), i32(L - 1u - p.step), d == 1u);
}
// gemm input [x_t ; h] for (direction, row)
@compute @workgroup_size(${WG}) fn gather(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= 2u * p.n * p.Kp) { return; }
  let dr = idx / p.Kp; let k = idx % p.Kp; let d = dr / p.n; let r = dr % p.n;
  var v = 0.0;
  if (k < p.In) {
    let t = frame(r, d);
    if (t >= 0) { v = xin[(r * p.Tmax + u32(t)) * p.In + k]; }
  } else if (k < p.In + p.H) { v = rs[p.oHS + dr * p.H + k - p.In]; }
  gin[d * p.pad1 + r * p.Kp + k] = v;
}
@compute @workgroup_size(${WG}) fn cell(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= 2u * p.n * p.H) { return; }
  let dr = idx / p.H; let u = idx % p.H; let d = dr / p.n; let r = dr % p.n;
  let t = frame(r, d);
  if (t < 0) { return; }
  let base = d * p.pad2 + r * 4u * p.H;
  let i = sigm(gates[base + u]); let f = sigm(gates[base + p.H + u]); let gg = tanh_s(gates[base + 2u * p.H + u]); let o = sigm(gates[base + 3u * p.H + u]);
  let cn = f * rs[p.oCS + idx] + i * gg;
  rs[p.oCS + idx] = cn;
  let hn = o * tanh_s(cn);
  rs[p.oHS + idx] = hn;
  xout[(r * p.Tmax + u32(t)) * 2u * p.H + d * p.H + u] = hn;
}
// the output layer + log-softmax per frame
@compute @workgroup_size(${WG}) fn project(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.Tmax) { return; }
  let r = idx / p.Tmax; let t = idx % p.Tmax;
  if (t >= lens[p.olen + r]) { return; }
  var mx = -1e30;
  for (var o = 0u; o < p.C; o++) {
    var acc = bias[o];
    for (var i = 0u; i < p.In; i++) { acc += wgt[o * p.In + i] * xin[(r * p.Tmax + t) * p.In + i]; }
    xout[idx * p.C + o] = acc;
    mx = max(mx, acc);
  }
  var z = 0.0;
  for (var o = 0u; o < p.C; o++) { z += exp(xout[idx * p.C + o] - mx); }
  let lz = mx + log(z);
  for (var o = 0u; o < p.C; o++) { xout[idx * p.C + o] -= lz; }
}
`;

export async function initGPU() {
  if (!globalThis.navigator?.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const want = { maxStorageBufferBindingSize: 64 << 20, maxBufferSize: 64 << 20, maxComputeWorkgroupsPerDimension: MAXWG };
  const requiredLimits = {};
  for (const [k, v] of Object.entries(want)) if (adapter.limits[k] >= v) requiredLimits[k] = v;
  const device = await adapter.requestDevice({ requiredLimits });
  return new GPU(device, adapter);
}

class GPU {
  constructor(device, adapter) {
    this.device = device; this.adapter = adapter;
    const layout = types => device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ...types.map((t, i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: t ? 'storage' : 'read-only-storage' } }))] });
    this.synthLayout = layout([1, 1, 0, 0, 1, 1, 1]); this.readerLayout = layout([0, 0, 0, 0, 1, 1, 1, 1]);
    this.errors = [];
    device.addEventListener('uncapturederror', e => { this.errors.push(e.error.message); console.error('WebGPU: ' + e.error.message); });
    const pipes = (code, names, bgl) => {
      const module = device.createShaderModule({ code }), pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      return Object.fromEntries(names.map(e => [e, device.createComputePipeline({ layout: pl, compute: { module, entryPoint: e } })]));
    };
    this.gemm = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: GEMM }), entryPoint: 'main' } });
    this.synthPipes = pipes(SYNTH, ['gather1', 'gather2', 'gather3', 'cell', 'window', 'attend', 'mixw', 'headin', 'sample', 'tick'], this.synthLayout);
    this.readerPipes = pipes(READER, ['conv', 'gather', 'cell', 'project'], this.readerLayout);
    const i = adapter.info || {};
    this.info = [i.vendor, i.architecture, i.description].filter(Boolean).join(' ') || 'WebGPU';
  }
  buf(sizeBytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, data = null) {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(sizeBytes / 16) * 16), usage });
    if (data) this.device.queue.writeBuffer(b, 0, data);
    return b;
  }
  f32(data) { return this.buf(data.byteLength, undefined, data); }
  zeros(n) { return this.buf(n * 4); }
  u32(arr) { return this.buf(arr.length * 4, undefined, Uint32Array.from(arr)); }
  /** A uniform buffer of 256-byte slots; each slot = [u32 fields..., f32 fields...]. */
  slots(list) {
    const ab = new ArrayBuffer(SLOT * list.length), dv = new DataView(ab);
    list.forEach(([u32s, f32s = []], s) => { u32s.forEach((v, i) => dv.setUint32(s * SLOT + 4 * i, v >>> 0, true)); f32s.forEach((v, i) => dv.setFloat32(s * SLOT + 4 * (u32s.length + i), v, true)); });
    return this.buf(ab.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Uint8Array(ab));
  }
  bindGroup(layout, uniform, buffers) {
    return this.device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: uniform, size: SLOT } },
      ...buffers.map((b, i) => ({ binding: i + 1, resource: b.buffer ? b : { buffer: b } }))] });
  }
  /** A GEMM job: X (rows x K) · W (K x N) + B -> Y; X and Y may be {buffer, offset, size}. */
  gemmJob(rows, K, N, X, W, B, Y) {
    const u = this.slots([[[rows, K, N, 0]]]);
    const res = b => b.buffer ? b : { buffer: b };
    const bg = this.device.createBindGroup({ layout: this.gemm.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: u, size: 16 } },
      { binding: 1, resource: res(X) }, { binding: 2, resource: res(W) }, { binding: 3, resource: res(B) }, { binding: 4, resource: res(Y) }] });
    return { bg, wg: [Math.ceil(N / COLS), Math.ceil(rows / TILE_ROWS)], u };
  }
  runGemm(pass, job) { pass.setPipeline(this.gemm); pass.setBindGroup(0, job.bg); pass.dispatchWorkgroups(...job.wg); }
  run(pass, pipe, bg, slot, count) {
    const wgs = Math.ceil(count / WG);
    pass.setPipeline(pipe); pass.setBindGroup(0, bg, [slot * SLOT]); pass.dispatchWorkgroups(Math.min(wgs, MAXWG), Math.ceil(wgs / MAXWG));
  }
  async read(buffer, bytes) {
    const rb = this.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, rb, 0, rb.size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = rb.getMappedRange().slice(0);
    rb.unmap(); rb.destroy();
    return out;
  }
}

/** W_ih (4H x In) and W_hh (4H x H) -> transposed [In+H (padded) x 4H] on the GPU, and the summed bias. */
function lstmWeights(gpu, ih, hh, bi, bh) {
  const [J, In] = ih.shape, H = hh.shape[1], Kp = padK(In + H);
  const W = new Float32Array(Kp * J), b = new Float32Array(J);
  for (let j = 0; j < J; j++) {
    for (let k = 0; k < In; k++) W[k * J + j] = ih.data[j * In + k];
    for (let k = 0; k < H; k++) W[(In + k) * J + j] = hh.data[j * H + k];
    b[j] = bi.data[j] + bh.data[j];
  }
  return { W: gpu.f32(W), B: gpu.f32(b), K: Kp, N: J, In, H };
}
function linearWeights(gpu, w, b) {   // (J x K) -> transposed (Kp x J)
  const [J, K] = w.shape, Kp = padK(K), W = new Float32Array(Kp * J);
  for (let j = 0; j < J; j++) for (let k = 0; k < K; k++) W[k * J + j] = w.data[j * K + k];
  return { W: gpu.f32(W), B: gpu.f32(b.data), K: Kp, N: J };
}

export class SynthGPU {
  constructor(gpu, gen) {
    this.gpu = gpu; this.H = gen.hidden; this.K = gen.K; this.M = gen.M; this.S = gen.style_dim;
    this.stats = gen.stats; this.writer = gen.writer;
    const sfx = n => n === 'lstm1' ? '' : '_l0';
    this.l = ['lstm1', 'lstm2', 'lstm3'].map(n => lstmWeights(gpu, gen.get(`${n}.weight_ih${sfx(n)}`), gen.get(`${n}.weight_hh${sfx(n)}`), gen.get(`${n}.bias_ih${sfx(n)}`), gen.get(`${n}.bias_hh${sfx(n)}`)));
    this.win = linearWeights(gpu, gen.get('window.weight'), gen.get('window.bias'));
    this.head = linearWeights(gpu, gen.get('head.weight'), gen.get('head.bias'));
    const st = gen.get('style.weight');
    this.style = gpu.f32(this.S ? st.data.slice(this.writer * this.S, (this.writer + 1) * this.S) : new Float32Array(4));
    this.lastRun = null;
  }

  /** Buffers, uniforms and bind groups for one run: n rows over the U characters of text. */
  prepare(text, n, { bias, seed, endFrac }, primeRows, stepsOut) {
    const { gpu, H, K, M, S } = this, C = N_CHARS, K1 = this.l[0].K, K2 = this.l[1].K, Kh = this.head.K;
    const ids = encodeText(text), U = ids.length;
    const lay = sizes => { const o = {}; let total = 0; for (const [k, v] of Object.entries(sizes)) { o[k] = total; total += align(v); } return [o, total]; };
    const [o, stTotal] = lay({ X: n * 3, W: n * C, H1: n * H, C1: n * H, H2: n * H, C2: n * H, H3: n * H, C3: n * H, KAPPA: n * K, PHI: n * (U + 1) });
    const [oi, giTotal] = lay({ IN1: n * K1, IN2: n * K2, HIN: n * Kh });
    const [oo, goTotal] = lay({ G: n * 4 * H, ABK: n * this.win.N, Y: n * this.head.N });
    Object.assign(o, oi, oo);
    const oDONE = 4, oIDS = 4 + align(n);
    const misc = new Uint32Array(oIDS + U); misc.set(ids, oIDS);
    const b = { misc: gpu.u32(misc), out: gpu.zeros(stepsOut * n * 3), prime: gpu.f32(primeRows && primeRows.length ? Float32Array.from(primeRows.flat()) : new Float32Array(4)),
                state: gpu.zeros(stTotal), gin: gpu.zeros(giTotal), gout: gpu.zeros(goTotal) };
    const base = [n, U, K, M, H, S, C, K1, K2, Kh, seed, 0, 0, stepsOut, o.X, o.W, o.H1, o.C1, o.H2, o.C2, o.H3, o.C3, o.IN1, o.IN2, o.G, o.ABK, o.KAPPA, o.PHI, o.Y, o.HIN, oDONE, oIDS];
    const slot = (mode, layer) => [base.map((v, i) => i === 11 ? mode : i === 12 ? layer : v), [bias, endFrac, this.stats[0], this.stats[1]]];
    const uni = gpu.slots([slot(0, 0), slot(1, 0), slot(0, 1), slot(0, 2), slot(0, 3)]);   // slots: sample, prime, cell 1-3
    const bg = gpu.bindGroup(gpu.synthLayout, uni, [b.misc, b.out, b.prime, this.style, b.state, b.gin, b.gout]);
    const region = (buf, k, floats) => ({ buffer: buf, offset: o[k] * 4, size: floats * 4 });
    const gem = [gpu.gemmJob(n, K1, 4 * H, region(b.gin, 'IN1', n * K1), this.l[0].W, this.l[0].B, region(b.gout, 'G', n * 4 * H)),
                 gpu.gemmJob(n, K2, 4 * H, region(b.gin, 'IN2', n * K2), this.l[1].W, this.l[1].B, region(b.gout, 'G', n * 4 * H)),
                 gpu.gemmJob(n, K2, 4 * H, region(b.gin, 'IN2', n * K2), this.l[2].W, this.l[2].B, region(b.gout, 'G', n * 4 * H)),
                 gpu.gemmJob(n, H, this.win.N, region(b.state, 'H1', n * H), this.win.W, this.win.B, region(b.gout, 'ABK', n * this.win.N)),   // X stride = H: its rows are the raw h1 state
                 gpu.gemmJob(n, Kh, this.head.N, region(b.gin, 'HIN', n * Kh), this.head.W, this.head.B, region(b.gout, 'Y', n * this.head.N))];
    return { b, bg, gem, uni, n, U, o, oDONE };
  }

  /** Records `steps` generator steps (mode 0 samples, mode 1 feeds the priming rows). */
  record(pass, S, steps, mode) {
    const { gpu, H } = this, P = gpu.synthPipes, { bg, n } = S;
    for (let s = 0; s < steps; s++) {
      gpu.run(pass, P.gather1, bg, 0, n * this.l[0].K);
      gpu.runGemm(pass, S.gem[0]);
      gpu.run(pass, P.cell, bg, 2, n * H);
      gpu.runGemm(pass, S.gem[3]);
      gpu.run(pass, P.window, bg, 0, n * this.K);
      gpu.run(pass, P.attend, bg, 0, n * (S.U + 1));
      gpu.run(pass, P.mixw, bg, 0, n * N_CHARS);
      gpu.run(pass, P.gather2, bg, 0, n * this.l[1].K);
      gpu.runGemm(pass, S.gem[1]);
      gpu.run(pass, P.cell, bg, 3, n * H);
      gpu.run(pass, P.gather3, bg, 0, n * this.l[2].K);
      gpu.runGemm(pass, S.gem[2]);
      gpu.run(pass, P.cell, bg, 4, n * H);
      gpu.run(pass, P.headin, bg, 0, n * this.head.K);
      gpu.runGemm(pass, S.gem[4]);
      gpu.run(pass, P.sample, bg, mode, n);
      gpu.run(pass, P.tick, bg, 0, 1);
    }
  }

  destroy(S) { for (const b of Object.values(S.b)) b.destroy(); S.uni.destroy(); for (const g of S.gem) g.u.destroy(); }

  submit(S, steps, mode) {
    const { device } = this.gpu, enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    this.record(pass, S, steps, mode);
    pass.end(); device.queue.submit([enc.finish()]);
  }

  /** Feeds rows (normalized) one per step in prime mode: the network sees rows[0..T-2], x ends as rows[T-1]. */
  feed(S, rows, block) {
    const { device } = this.gpu, xs = new Float32Array(S.n * 3);
    for (let r = 0; r < S.n; r++) xs.set(rows[0], 3 * r);
    device.queue.writeBuffer(S.b.state, S.o.X * 4, xs);
    for (let s = 0; s < rows.length - 1; s += block) this.submit(S, Math.min(block, rows.length - 1 - s), 1);
    device.queue.writeBuffer(S.b.misc, 0, new Uint32Array([0]));
  }

  /** Same contract as SynthCPU.sampleN; prime = {text, ink}. */
  async sampleN(text, n, { bias = 1.0, seed = 0, prime = null, maxSteps = 0, endFrac = 0.5, block = 48 } = {}) {
    const t0 = performance.now(), { gpu } = this, [sx, sy] = this.stats;
    const full = prime ? prime.text + ' ' + text : text, U = encodeText(full).length;
    const limit = maxSteps || 40 * U + 100;
    const rows = [];
    if (prime) {
      const p = prime.ink, T = p.length / 3;
      for (let t = 0; t < T - 1; t++) rows.push([p[3 * t] / sx, p[3 * t + 1] / sy, p[3 * t + 2]]);
      rows.push([p[3 * (T - 1)] / sx, p[3 * (T - 1) + 1] / sy, 1]);
    } else rows.push([0, 0, 0]);
    const S = this.prepare(full, n, { bias, seed, endFrac }, rows, limit);
    this.feed(S, rows, block);
    let steps = 0, done = null;
    while (steps < limit) {
      const k = Math.min(block, limit - steps);
      this.submit(S, k, 0);
      steps += k;
      done = new Uint32Array(await gpu.read(S.b.misc, (S.oDONE + n) * 4)).subarray(S.oDONE, S.oDONE + n);
      if (done.every(d => d > 0)) break;
    }
    const out = new Float32Array(await gpu.read(S.b.out, steps * n * 3 * 4));
    const inks = [];
    for (let r = 0; r < n; r++) {
      const T = done[r] > 0 ? done[r] : steps, ink = new Float32Array((T + 1) * 3);
      for (let t = 0; t < T; t++) { const ob = (t * n + r) * 3; ink[3 * (t + 1)] = out[ob]; ink[3 * (t + 1) + 1] = out[ob + 1]; ink[3 * (t + 1) + 2] = out[ob + 2]; }
      ink[2] = 1; ink[ink.length - 1] = 1;
      inks.push(ink);
    }
    this.destroy(S);
    this.lastRun = { steps: steps + rows.length - 1, ms: performance.now() - t0, n };
    return inks;
  }

  /** For tests: teacher-forced rows (after optional priming rows) -> head output and phi per step. */
  async replay(text, inputs, { primeText = null, primeRows = null } = {}) {
    const full = primeText ? primeText + ' ' + text : text, pre = primeRows ? primeRows.length : 0;
    const rows = [...(primeRows || []), ...inputs].map(r => Array.from(r));
    rows.push([0, 0, 0]);
    const S = this.prepare(full, 1, { bias: 1, seed: 0, endFrac: 0.5 }, rows, 1), { gpu } = this, ys = [], phis = [];
    const xs = Float32Array.from(rows[0]);
    gpu.device.queue.writeBuffer(S.b.state, S.o.X * 4, xs);
    for (let s = 0; s < rows.length - 1; s++) {
      this.submit(S, 1, 1);
      if (s >= pre) {
        ys.push(Array.from(new Float32Array(await gpu.read(S.b.gout, (S.o.Y + this.head.N) * 4)).subarray(S.o.Y, S.o.Y + this.head.N)));
        phis.push(Array.from(new Float32Array(await gpu.read(S.b.state, (S.o.PHI + S.U + 1) * 4)).subarray(S.o.PHI, S.o.PHI + S.U + 1)));
      }
    }
    this.destroy(S);
    return { y: ys, phi: phis };
  }
}

export class ReaderGPU {
  constructor(gpu, rd) {
    this.gpu = gpu; this.H = rd.cfg.hidden; this.layers = rd.cfg.layers;
    this.conv = [0, 2, 4].map(i => { const w = rd.get(`conv.${i}.weight`); return { W: gpu.f32(w.data), B: gpu.f32(rd.get(`conv.${i}.bias`).data), Cout: w.shape[0], Cin: w.shape[1] }; });
    this.lstm = [];
    for (let l = 0; l < this.layers; l++) this.lstm.push(['', '_reverse'].map(sfx => lstmWeights(gpu, rd.get(`lstm.weight_ih_l${l}${sfx}`), rd.get(`lstm.weight_hh_l${l}${sfx}`), rd.get(`lstm.bias_ih_l${l}${sfx}`), rd.get(`lstm.bias_hh_l${l}${sfx}`))));
    const ow = rd.get('out.weight');
    this.out = { W: gpu.f32(ow.data), B: gpu.f32(rd.get('out.bias').data), In: ow.shape[1] };
    this.lastRun = null;
  }

  async logProbsBatch(inks) {
    const { gpu, H } = this, { device } = gpu, P = gpu.readerPipes, n = inks.length, t0 = performance.now();
    const feats = inks.map(readerFeats), N = inks.map(i => i.length / 3), Nmax = Math.max(...N);
    const L2 = N.map(v => Math.floor((v + 1) / 2)), L3 = L2.map(v => Math.floor((v + 1) / 2));
    const T1 = Nmax, T2 = Math.floor((Nmax + 1) / 2), T3 = Math.floor((T2 + 1) / 2), Tmax = T3;
    const x0 = new Float32Array(n * Nmax * 5);
    feats.forEach((f, r) => x0.set(f, r * Nmax * 5));
    const Kmax = Math.max(...this.lstm.map(l => l[0].K)), oHS = 0, oCS = align(2 * n * H);
    const b = { x: [gpu.f32(x0), gpu.zeros(n * T1 * 64), gpu.zeros(n * T2 * 128), gpu.zeros(n * T3 * 128)], lens: gpu.u32([...N, ...N, ...L2, ...L3]),
                rs: gpu.zeros(2 * align(2 * n * H)), gin: gpu.zeros(2 * align(n * Kmax)), gates: gpu.zeros(2 * align(n * 4 * H)),
                seq: [gpu.zeros(n * Tmax * 2 * H), gpu.zeros(n * Tmax * 2 * H)], lp: gpu.zeros(n * Tmax * N_CHARS), dummy: gpu.zeros(4) };
    // uniform slots: conv 0-2, then one per layer and step, then project
    const u = (Tm, Tin, Cin, Cout, stride, In, Kp, step, olen, olenIn) => [[n, Tm, Tin, Cin, Cout, stride, H, In, Kp, step, N_CHARS, olen, olenIn, oHS, oCS, align(n * Kmax), align(n * 4 * H), 0, 0, 0]];
    const slots = [u(T1, Nmax, 5, 64, 1, 0, 0, 0, n, 0), u(T2, T1, 64, 128, 2, 0, 0, 0, 2 * n, n), u(T3, T2, 128, 128, 2, 0, 0, 0, 3 * n, 2 * n)];
    const layerSlot = [];
    for (let l = 0; l < this.layers; l++) {
      const In = l ? 2 * H : 128, Kp = this.lstm[l][0].K;
      layerSlot.push(slots.length);
      for (let s = 0; s < Tmax; s++) slots.push(u(Tmax, Tmax, 0, 0, 1, In, Kp, s, 3 * n, 3 * n));
    }
    const projSlot = slots.length;
    slots.push(u(Tmax, Tmax, 0, 0, 1, this.out.In, 0, 0, 3 * n, 3 * n));
    const uni = gpu.slots(slots);
    const bind = (xin, wgt, bias, xout) => gpu.bindGroup(gpu.readerLayout, uni, [b.lens, xin, wgt, bias, xout, b.rs, b.gin, b.gates]);
    const enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    [[b.x[0], b.x[1], T1], [b.x[1], b.x[2], T2], [b.x[2], b.x[3], T3]].forEach(([xin, xout, T], i) => {
      const c = this.conv[i];
      gpu.run(pass, P.conv, bind(xin, c.W, c.B, xout), i, n * T * c.Cout);
    });
    pass.end();
    let xin = b.x[3];
    const jobs = [];
    for (let l = 0; l < this.layers; l++) {
      const Kp = this.lstm[l][0].K, seq = b.seq[l % 2];
      const bg = bind(xin, b.dummy, b.dummy, seq);
      const gems = this.lstm[l].map((w, d) => gpu.gemmJob(n, Kp, 4 * H, { buffer: b.gin, offset: align(n * Kmax) * d * 4, size: n * Kp * 4 }, w.W, w.B,
                                                          { buffer: b.gates, offset: align(n * 4 * H) * d * 4, size: n * 4 * H * 4 }));
      jobs.push(...gems);
      enc.clearBuffer(b.rs);                       // fresh h and c for this layer
      pass = enc.beginComputePass();
      for (let s = 0; s < Tmax; s++) {
        gpu.run(pass, P.gather, bg, layerSlot[l] + s, 2 * n * Kp);
        for (const g of gems) gpu.runGemm(pass, g);
        gpu.run(pass, P.cell, bg, layerSlot[l] + s, 2 * n * H);
      }
      pass.end();
      xin = seq;
    }
    pass = enc.beginComputePass();
    gpu.run(pass, P.project, bind(xin, this.out.W, this.out.B, b.lp), projSlot, n * Tmax);
    pass.end();
    device.queue.submit([enc.finish()]);
    const lp = new Float32Array(await gpu.read(b.lp, n * Tmax * N_CHARS * 4));
    for (const v of Object.values(b)) (Array.isArray(v) ? v : [v]).forEach(x => x.destroy());
    uni.destroy(); for (const j of jobs) j.u.destroy();
    this.lastRun = { ms: performance.now() - t0, n };
    return inks.map((_, r) => ({ lp: lp.slice(r * Tmax * N_CHARS, (r * Tmax + L3[r]) * N_CHARS), T: L3[r] }));
  }
}
