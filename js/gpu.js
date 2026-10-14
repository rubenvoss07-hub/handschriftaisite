// The generator and the readers on the GPU through WebGPU: the same math as synth_cpu.js / reader_cpu.js,
// batched over the candidates. A line is about 1300 sequential steps, so the cost is per step and 64
// candidates cost about what one does. A generator step is 7 dispatches (each LSTM layer is one fused
// kernel: input assembly, gates, cell), a reader frame is 1. Steps are recorded in blocks into one
// command buffer; between blocks the host reads back only the done flags, and the points once at the end.
import { encodeText, N_CHARS } from './vocab.js';
import { readerFeats } from './reader_cpu.js';

const WG = 64, MAXWG = 65535, SLOT = 256, MAXU = 512, MAXN = 64;
const TK = 8, TKL = 32;                               // k per tile: fused LSTM (W tile shared), linear
const padTo = (k, m) => Math.ceil(k / m) * m;
const align = n => Math.ceil(n / 64) * 64;           // float offsets at 256-byte boundaries
const tiling = n => n <= 16 ? [n, 1] : [16, Math.ceil(n / 16)];   // [rows per thread, row groups]

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

// The fused LSTM step and the linear layer, shared by the generator and the readers; XVAL supplies the
// input element (r, k) and the caller's uniform names K, Kp, N, the weight offsets and the h/c offsets.
// Weights: W[(k) * N + u * 4 + g] (unit-major gates) with bias[u * 4 + g]; linear: W[k * N + j], bias[j].
const tiled = (T, RG) => /* wgsl */`
var<workgroup> wt: array<f32, ${TK * 256}>;
var<workgroup> xs: array<f32, ${TK * MAXN}>;
var<workgroup> xl: array<f32, ${TKL * MAXN}>;
@compute @workgroup_size(64, ${RG}) fn lstm(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let d = wg.z;
  let u = wg.x * 64u + li.x;
  let tid = li.y * 64u + li.x;
  let r0 = li.y * ${T}u;
  let oWt = wsel(d); let oB = bsel(d);
  var acc: array<f32, ${4 * T}>;
  for (var i = 0u; i < ${4 * T}u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < p.gKp; k0 += ${TK}u) {
    for (var e = tid; e < ${TK * 256}u; e += ${64 * RG}u) {
      let kk = e / 256u; let col = wg.x * 256u + e % 256u;
      wt[e] = select(0.0, wts[oWt + (k0 + kk) * p.gN + col], col < p.gN);
    }
    for (var e = tid; e < ${TK}u * p.n; e += ${64 * RG}u) {
      xs[e] = xval(d, e / ${TK}u, k0 + e % ${TK}u);
    }
    workgroupBarrier();
    if (u < p.H) {
      for (var kk = 0u; kk < ${TK}u; kk++) {
        let wb = kk * 256u + li.x * 4u;
        let w0 = wt[wb]; let w1 = wt[wb + 1u]; let w2 = wt[wb + 2u]; let w3 = wt[wb + 3u];
        for (var rr = 0u; rr < ${T}u; rr++) {
          let x = xs[(r0 + rr) * ${TK}u + kk];
          acc[rr * 4u] += w0 * x; acc[rr * 4u + 1u] += w1 * x; acc[rr * 4u + 2u] += w2 * x; acc[rr * 4u + 3u] += w3 * x;
        }
      }
    }
    workgroupBarrier();
  }
  if (u < p.H) {
    for (var rr = 0u; rr < ${T}u; rr++) {
      let r = r0 + rr;
      if (r < p.n) {
        let i = sigm(acc[rr * 4u] + wts[oB + u * 4u]); let f = sigm(acc[rr * 4u + 1u] + wts[oB + u * 4u + 1u]);
        let g = tanh_s(acc[rr * 4u + 2u] + wts[oB + u * 4u + 2u]); let o = sigm(acc[rr * 4u + 3u] + wts[oB + u * 4u + 3u]);
        cell_out(d, r, u, i, f, g, o);
      }
    }
  }
}
@compute @workgroup_size(64, ${RG}) fn linear(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let j = wg.x * 64u + li.x;
  let tid = li.y * 64u + li.x;
  let r0 = li.y * ${T}u;
  var acc: array<f32, ${T}>;
  for (var i = 0u; i < ${T}u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < p.gKp; k0 += ${TKL}u) {
    for (var e = tid; e < ${TKL}u * p.n; e += ${64 * RG}u) {
      xl[e] = xval(0u, e / ${TKL}u, k0 + e % ${TKL}u);
    }
    workgroupBarrier();
    if (j < p.gN) {
      for (var kk = 0u; kk < ${TKL}u; kk++) {
        let w = wts[p.oWt + (k0 + kk) * p.gN + j];
        for (var rr = 0u; rr < ${T}u; rr++) { acc[rr] += w * xl[(r0 + rr) * ${TKL}u + kk]; }
      }
    }
    workgroupBarrier();
  }
  if (j < p.gN) {
    for (var rr = 0u; rr < ${T}u; rr++) {
      let r = r0 + rr;
      if (r < p.n) { gout[p.oOut + r * p.gN + j] = acc[rr] + wts[p.oB + j]; }
    }
  }
}`;

// ------------------------------------------------------------------ generator
const SYNTH_HEAD = /* wgsl */`
${COMMON}
struct P { n: u32, U: u32, K: u32, M: u32, H: u32, S: u32, C: u32, seed: u32, mode: u32, steps_out: u32,
           oX: u32, oW: u32, oH1p: u32, oH1n: u32, oC1: u32, oH2p: u32, oH2n: u32, oC2: u32, oH3p: u32, oH3n: u32, oC3: u32,
           oKAPPA: u32, oPHI: u32, oABK: u32, oY: u32, oDONE: u32, oIDS: u32,
           gmode: u32, gK: u32, gKp: u32, gN: u32, oWt: u32, oB: u32, oOut: u32, oH: u32, oC: u32,
           bias: f32, end_frac: f32, sx: f32, sy: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> misc: array<u32>;   // [0] = step, done (n) at oDONE, ids (U) at oIDS
@group(0) @binding(2) var<storage, read_write> out: array<f32>;    // steps_out x n x 3
@group(0) @binding(3) var<storage, read> prime: array<f32>;        // rows to feed in mode 1
@group(0) @binding(4) var<storage, read> style: array<f32>;        // S
@group(0) @binding(5) var<storage, read_write> st: array<f32>;     // the recurrent state
@group(0) @binding(6) var<storage, read_write> gout: array<f32>;   // window and head outputs
@group(0) @binding(7) var<storage, read> wts: array<f32>;          // all weights

// the input element k of row r for the layer being computed: 1 = [x, style, w, h1], 2 = [x, style, h1, w, h2],
// 3 = [x, style, h2, w, h3], 4 = [h1], 5 = [h1, h2, h3]; h of the own layer is the previous step's
fn xval(d: u32, r: u32, k: u32) -> f32 {
  if (k >= p.gK) { return 0.0; }
  let m = p.gmode;
  if (m == 4u) { return st[p.oH1n + r * p.H + k]; }
  if (m == 5u) {
    if (k < p.H) { return st[p.oH1n + r * p.H + k]; }
    if (k < 2u * p.H) { return st[p.oH2n + r * p.H + k - p.H]; }
    return st[p.oH3n + r * p.H + k - 2u * p.H];
  }
  if (k < 3u) { return st[p.oX + r * 3u + k]; }
  if (k < 3u + p.S) { return style[k - 3u]; }
  var kk = k - 3u - p.S;
  if (m == 1u) {
    if (kk < p.C) { return st[p.oW + r * p.C + kk]; }
    return st[p.oH1p + r * p.H + kk - p.C];
  }
  let below = select(p.oH2n, p.oH1n, m == 2u);
  if (kk < p.H) { return st[below + r * p.H + kk]; }
  kk -= p.H;
  if (kk < p.C) { return st[p.oW + r * p.C + kk]; }
  return st[p.oH + r * p.H + kk - p.C];
}
fn wsel(d: u32) -> u32 { return p.oWt; }
fn bsel(d: u32) -> u32 { return p.oB; }
fn cell_out(d: u32, r: u32, u: u32, i: f32, f: f32, g: f32, o: f32) {
  let idx = r * p.H + u;
  let cn = f * st[p.oC + idx] + i * g;
  st[p.oC + idx] = cn;
  st[p.oOut + idx] = o * tanh_s(cn);
}
`;
const SYNTH_TAIL = /* wgsl */`
var<workgroup> ka: array<f32, 32>;
var<workgroup> aa: array<f32, 32>;
var<workgroup> bb: array<f32, 32>;
var<workgroup> ph: array<f32, ${MAXU}>;
// the window (one workgroup per row): kappa moves forward, phi over the characters plus one past the end,
// w = the mix of one-hots
@compute @workgroup_size(64) fn attendw(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x; let t = li.x; let K = p.K;
  if (t < K) {
    let base = p.oABK + r * 3u * K;
    aa[t] = exp(gout[base + t]); bb[t] = exp(gout[base + K + t]);
    let kp = st[p.oKAPPA + r * K + t] + exp(gout[base + 2u * K + t]) / 25.0;
    ka[t] = kp; st[p.oKAPPA + r * K + t] = kp;
  }
  workgroupBarrier();
  for (var u = t; u <= p.U; u += 64u) {
    var s = 0.0;
    for (var j = 0u; j < K; j++) { let d = ka[j] - f32(u); s += aa[j] * exp(-bb[j] * d * d); }
    ph[u] = s; st[p.oPHI + r * (p.U + 1u) + u] = s;
  }
  workgroupBarrier();
  for (var c = t; c < p.C; c += 64u) {
    var s = 0.0;
    if (c != 0u) { for (var u = 0u; u < p.U; u++) { if (misc[p.oIDS + u] == c) { s += ph[u]; } } }
    st[p.oW + r * p.C + c] = s;
  }
}
// draw the next point from the mixture (mode 0) or feed the next priming row (mode 1); one workgroup for
// all rows, so the last thread can count the step
@compute @workgroup_size(64) fn sample(@builtin(local_invocation_id) li: vec3u) {
  let r = li.x; let step = misc[0];
  if (r < p.n) {
    if (p.mode == 1u) {
      let nb = (step + 1u) * 3u;
      st[p.oX + r * 3u] = prime[nb]; st[p.oX + r * 3u + 1u] = prime[nb + 1u]; st[p.oX + r * 3u + 2u] = prime[nb + 2u];
    } else {
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
  }
  workgroupBarrier();
  if (li.x == 0u) { misc[0] = step + 1u; }
}
`;

// ------------------------------------------------------------------ readers
const READER_HEAD = /* wgsl */`
${COMMON}
struct P { n: u32, Tmax: u32, Tin: u32, Cin: u32, Cout: u32, stride: u32, H: u32, In: u32, step: u32, C: u32,
           olen: u32, olen_in: u32, oHSp: u32, oHSn: u32, oCS: u32, gK: u32, gKp: u32, gN: u32,
           oW0: u32, oB0: u32, oW1: u32, oB1: u32, oWt: u32, oB: u32, oOut: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> lens: array<u32>;         // valid frames per row, per stage
@group(0) @binding(2) var<storage, read> xin: array<f32>;          // n x Tin x Cin  (lstm: n x Tmax x In)
@group(0) @binding(3) var<storage, read_write> xout: array<f32>;   // n x Tmax x Cout (lstm: n x Tmax x 2H; project: n x Tmax x C)
@group(0) @binding(4) var<storage, read_write> rs: array<f32>;     // h (two copies) and c per (direction, row)
@group(0) @binding(5) var<storage, read> wts: array<f32>;
@group(0) @binding(6) var<storage, read_write> gout: array<f32>;   // unused by the readers (the linear entry writes xout)
@group(0) @binding(7) var<storage, read_write> misc: array<u32>;   // unused

// frame of row r at this step for direction d: forward s, backward len-1-s; -1 when the row is done
fn frame(r: u32, d: u32) -> i32 {
  let L = lens[p.olen + r];
  if (p.step >= L) { return -1; }
  return select(i32(p.step), i32(L - 1u - p.step), d == 1u);
}
fn xval(d: u32, r: u32, k: u32) -> f32 {
  if (k < p.In) {
    let t = frame(r, d);
    if (t < 0) { return 0.0; }
    return xin[(r * p.Tmax + u32(t)) * p.In + k];
  }
  if (k < p.gK) { return rs[p.oHSp + (d * p.n + r) * p.H + k - p.In]; }
  return 0.0;
}
fn wsel(d: u32) -> u32 { return select(p.oW0, p.oW1, d == 1u); }
fn bsel(d: u32) -> u32 { return select(p.oB0, p.oB1, d == 1u); }
fn cell_out(d: u32, r: u32, u: u32, i: f32, f: f32, g: f32, o: f32) {
  let t = frame(r, d);
  let idx = (d * p.n + r) * p.H + u;
  if (t < 0) { rs[p.oHSn + idx] = rs[p.oHSp + idx]; return; }
  let cn = f * rs[p.oCS + idx] + i * g;
  rs[p.oCS + idx] = cn;
  let hn = o * tanh_s(cn);
  rs[p.oHSn + idx] = hn;
  xout[(r * p.Tmax + u32(t)) * 2u * p.H + d * p.H + u] = hn;
}
`;
const READER_TAIL = /* wgsl */`
// conv1d, kernel 5, padding 2, ReLU; positions past the row's own length count as zero
@compute @workgroup_size(${WG}) fn conv(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.Tmax * p.Cout) { return; }
  let o = idx % p.Cout; let t = (idx / p.Cout) % p.Tmax; let r = idx / (p.Cout * p.Tmax);
  if (t >= lens[p.olen + r]) { xout[idx] = 0.0; return; }
  var acc = wts[p.oB0 + o];
  for (var j = 0u; j < 5u; j++) {
    let ti = i32(t * p.stride + j) - 2;
    if (ti < 0 || u32(ti) >= lens[p.olen_in + r]) { continue; }
    let xb = (r * p.Tin + u32(ti)) * p.Cin;
    for (var i = 0u; i < p.Cin; i++) { acc += wts[p.oW0 + (o * p.Cin + i) * 5u + j] * xin[xb + i]; }
  }
  xout[idx] = max(acc, 0.0);
}
// the output layer + log-softmax per frame
@compute @workgroup_size(${WG}) fn project(@builtin(global_invocation_id) g: vec3u) {
  let idx = gidx(g); if (idx >= p.n * p.Tmax) { return; }
  let r = idx / p.Tmax; let t = idx % p.Tmax;
  if (t >= lens[p.olen + r]) { return; }
  var mx = -1e30;
  for (var o = 0u; o < p.C; o++) {
    var acc = wts[p.oB0 + o];
    for (var i = 0u; i < p.In; i++) { acc += wts[p.oW0 + o * p.In + i] * xin[(r * p.Tmax + t) * p.In + i]; }
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
    this.synthLayout = layout([1, 1, 0, 0, 1, 1, 0]); this.readerLayout = layout([0, 0, 1, 1, 0, 1, 1]);
    this.synthPL = device.createPipelineLayout({ bindGroupLayouts: [this.synthLayout] });
    this.readerPL = device.createPipelineLayout({ bindGroupLayouts: [this.readerLayout] });
    const pipes = (code, names, pl) => {
      const module = device.createShaderModule({ code });
      return Object.fromEntries(names.map(e => [e, device.createComputePipeline({ layout: pl, compute: { module, entryPoint: e } })]));
    };
    this.synthPipes = pipes(SYNTH_HEAD + SYNTH_TAIL, ['attendw', 'sample'], this.synthPL);
    this.readerPipes = pipes(READER_HEAD + READER_TAIL, ['conv', 'project'], this.readerPL);
    this.tiledCache = new Map();
    this.errors = [];
    device.addEventListener('uncapturederror', e => { this.errors.push(e.error.message); console.error('WebGPU: ' + e.error.message); });
    const i = adapter.info || {};
    this.info = [i.vendor, i.architecture, i.description].filter(Boolean).join(' ') || 'WebGPU';
  }
  /** The fused LSTM / linear pipelines for n rows (rows per thread and row groups depend on n). */
  tiledPipes(kind, n) {
    const [T, RG] = tiling(n), key = `${kind}:${T}x${RG}`;
    if (!this.tiledCache.has(key)) {
      const head = kind === 'synth' ? SYNTH_HEAD : READER_HEAD, pl = kind === 'synth' ? this.synthPL : this.readerPL;
      const module = this.device.createShaderModule({ code: head + tiled(T, RG) });
      this.tiledCache.set(key, { lstm: this.device.createComputePipeline({ layout: pl, compute: { module, entryPoint: 'lstm' } }),
                                 linear: this.device.createComputePipeline({ layout: pl, compute: { module, entryPoint: 'linear' } }), RG });
    }
    return this.tiledCache.get(key);
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
      ...buffers.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))] });
  }
  dispatch(pass, pipe, bg, slot, x, y = 1, z = 1) {
    pass.setPipeline(pipe); pass.setBindGroup(0, bg, [slot * SLOT]); pass.dispatchWorkgroups(x, y, z);
  }
  run(pass, pipe, bg, slot, count) {
    const wgs = Math.ceil(count / WG);
    this.dispatch(pass, pipe, bg, slot, Math.min(wgs, MAXWG), Math.ceil(wgs / MAXWG));
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

/** Packs weights into one Float32Array; add() returns the float offset of what it appended. */
class Pack {
  constructor() { this.parts = []; this.size = 0; }
  add(a) { const o = this.size; this.parts.push(a); this.size += align(a.length); return o; }
  /** W_ih (4H x In), W_hh (4H x H) -> [Kp x 4H] unit-major (u * 4 + gate) and bias[u * 4 + gate]. */
  lstm(ih, hh, bi, bh) {
    const [J, In] = ih.shape, H = hh.shape[1], K = In + H, Kp = padTo(K, TK), W = new Float32Array(Kp * J), b = new Float32Array(J);
    for (let g = 0; g < 4; g++) for (let u = 0; u < H; u++) {
      const row = g * H + u, col = u * 4 + g;
      for (let k = 0; k < In; k++) W[k * J + col] = ih.data[row * In + k];
      for (let k = 0; k < H; k++) W[(In + k) * J + col] = hh.data[row * H + k];
      b[col] = bi.data[row] + bh.data[row];
    }
    return { oW: this.add(W), oB: this.add(b), K, Kp, N: J, In, H };
  }
  /** (J x K) -> [Kp x J] and the bias. */
  linear(w, b) {
    const [J, K] = w.shape, Kp = padTo(K, TKL), W = new Float32Array(Kp * J);
    for (let j = 0; j < J; j++) for (let k = 0; k < K; k++) W[k * J + j] = w.data[j * K + k];
    return { oW: this.add(W), oB: this.add(b.data), K, Kp, N: J };
  }
  upload(gpu) {
    const all = new Float32Array(this.size);
    let o = 0;
    for (const p of this.parts) { all.set(p, o); o += align(p.length); }
    return gpu.f32(all);
  }
}

export class SynthGPU {
  constructor(gpu, gen) {
    this.gpu = gpu; this.H = gen.hidden; this.K = gen.K; this.M = gen.M; this.S = gen.style_dim;
    this.stats = gen.stats; this.writer = gen.writer;
    const pk = new Pack(), sfx = n => n === 'lstm1' ? '' : '_l0';
    this.l = ['lstm1', 'lstm2', 'lstm3'].map(n => pk.lstm(gen.get(`${n}.weight_ih${sfx(n)}`), gen.get(`${n}.weight_hh${sfx(n)}`), gen.get(`${n}.bias_ih${sfx(n)}`), gen.get(`${n}.bias_hh${sfx(n)}`)));
    this.win = pk.linear(gen.get('window.weight'), gen.get('window.bias'));
    this.head = pk.linear(gen.get('head.weight'), gen.get('head.bias'));
    this.wts = pk.upload(gpu);
    const st = gen.get('style.weight');
    this.style = gpu.f32(this.S ? st.data.slice(this.writer * this.S, (this.writer + 1) * this.S) : new Float32Array(4));
    this.lastRun = null;
  }

  /** Buffers, uniform slots and the bind group for one run: n rows over the U characters of text. */
  prepare(text, n, { bias, seed, endFrac }, primeRows, stepsOut) {
    if (n > MAXN) throw new Error(`at most ${MAXN} candidates at once`);
    const { gpu, H, K, M, S } = this, C = N_CHARS;
    const ids = encodeText(text), U = ids.length;
    if (U + 1 > MAXU) throw new Error('line too long');
    const lay = sizes => { const o = {}; let total = 0; for (const [k, v] of Object.entries(sizes)) { o[k] = total; total += align(v); } return [o, total]; };
    const [o, stTotal] = lay({ X: n * 3, W: n * C, H1a: n * H, H1b: n * H, C1: n * H, H2a: n * H, H2b: n * H, C2: n * H, H3a: n * H, H3b: n * H, C3: n * H,
                               KAPPA: n * K, PHI: n * (U + 1) });
    const [og, goTotal] = lay({ ABK: n * this.win.N, Y: n * this.head.N });
    const oDONE = 4, oIDS = 4 + align(n);
    const misc = new Uint32Array(oIDS + U); misc.set(ids, oIDS);
    const b = { misc: gpu.u32(misc), out: gpu.zeros(stepsOut * n * 3), prime: gpu.f32(primeRows && primeRows.length ? Float32Array.from(primeRows.flat()) : new Float32Array(4)),
                state: gpu.zeros(stTotal), gout: gpu.zeros(goTotal) };
    // slot fields: the base part, then the gemm part; parity q says which h copy is the previous step's
    const base = (mode, q) => [n, U, K, M, H, S, C, seed, mode, stepsOut, o.X, o.W, q ? o.H1b : o.H1a, q ? o.H1a : o.H1b, o.C1,
                               q ? o.H2b : o.H2a, q ? o.H2a : o.H2b, o.C2, q ? o.H3b : o.H3a, q ? o.H3a : o.H3b, o.C3, o.KAPPA, o.PHI, og.ABK, og.Y, oDONE, oIDS];
    const f = [bias, endFrac, this.stats[0], this.stats[1]];
    const gem = (q, gmode, w, oOut, oH, oC) => [[...base(0, q), gmode, w.K, w.Kp, w.N, w.oW, w.oB, oOut, oH, oC], f];
    const slots = [[[...base(0, 0), 0, 0, 0, 0, 0, 0, 0, 0, 0], f], [[...base(1, 0), 0, 0, 0, 0, 0, 0, 0, 0, 0], f]];
    for (const q of [0, 1]) {
      const hn = l => q ? [o.H1a, o.H2a, o.H3a][l] : [o.H1b, o.H2b, o.H3b][l], hp = l => q ? [o.H1b, o.H2b, o.H3b][l] : [o.H1a, o.H2a, o.H3a][l];
      slots.push(gem(q, 1, this.l[0], hn(0), hp(0), o.C1), gem(q, 4, this.win, og.ABK, 0, 0), gem(q, 2, this.l[1], hn(1), hp(1), o.C2),
                 gem(q, 3, this.l[2], hn(2), hp(2), o.C3), gem(q, 5, this.head, og.Y, 0, 0));
    }
    const uni = gpu.slots(slots);
    const bg = gpu.bindGroup(gpu.synthLayout, uni, [b.misc, b.out, b.prime, this.style, b.state, b.gout, this.wts]);
    return { b, bg, uni, n, U, o, og, oDONE, parity: 0, pipes: gpu.tiledPipes('synth', n) };
  }

  /** Records `steps` generator steps (mode 0 samples, mode 1 feeds the priming rows). */
  record(pass, S, steps, mode) {
    const { gpu, H } = this, P = gpu.synthPipes, { bg, n, pipes } = S, RG = pipes.RG;
    const lstm = slot => gpu.dispatch(pass, pipes.lstm, bg, slot, Math.ceil(H / 64), 1, 1);
    const linear = (slot, N) => gpu.dispatch(pass, pipes.linear, bg, slot, Math.ceil(N / 64), 1, 1);
    for (let s = 0; s < steps; s++) {
      const q = S.parity, s0 = 2 + q * 5;
      lstm(s0);                                  // layer 1
      linear(s0 + 1, this.win.N);                // window
      gpu.dispatch(pass, P.attendw, bg, 0, n);   // kappa, phi, w
      lstm(s0 + 2);                              // layer 2
      lstm(s0 + 3);                              // layer 3
      linear(s0 + 4, this.head.N);               // head
      gpu.dispatch(pass, P.sample, bg, mode, 1); // draw (or feed), count the step
      S.parity ^= 1;
    }
  }

  destroy(S) { for (const b of Object.values(S.b)) b.destroy(); S.uni.destroy(); }

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
  async sampleN(text, n, { bias = 1.0, seed = 0, prime = null, maxSteps = 0, endFrac = 0.5, block = 64 } = {}) {
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
    gpu.device.queue.writeBuffer(S.b.state, S.o.X * 4, Float32Array.from(rows[0]));
    for (let s = 0; s < rows.length - 1; s++) {
      this.submit(S, 1, 1);
      if (s >= pre) {
        ys.push(Array.from(new Float32Array(await gpu.read(S.b.gout, (S.og.Y + this.head.N) * 4)).subarray(S.og.Y, S.og.Y + this.head.N)));
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
    const pk = new Pack();
    this.conv = [0, 2, 4].map(i => { const w = rd.get(`conv.${i}.weight`); return { oW: pk.add(w.data), oB: pk.add(rd.get(`conv.${i}.bias`).data), Cout: w.shape[0], Cin: w.shape[1] }; });
    this.lstm = [];
    for (let l = 0; l < this.layers; l++) this.lstm.push(['', '_reverse'].map(sfx => pk.lstm(rd.get(`lstm.weight_ih_l${l}${sfx}`), rd.get(`lstm.weight_hh_l${l}${sfx}`), rd.get(`lstm.bias_ih_l${l}${sfx}`), rd.get(`lstm.bias_hh_l${l}${sfx}`))));
    const ow = rd.get('out.weight');
    this.out = { oW: pk.add(ow.data), oB: pk.add(rd.get('out.bias').data), In: ow.shape[1] };
    this.wts = pk.upload(gpu);
    this.lastRun = null;
  }

  async logProbsBatch(inks) {
    const { gpu, H } = this, { device } = gpu, P = gpu.readerPipes, t0 = performance.now(), outs = [];
    for (let s = 0; s < inks.length; s += MAXN) outs.push(...await this.batch(inks.slice(s, s + MAXN)));
    this.lastRun = { ms: performance.now() - t0, n: inks.length };
    return outs;
  }

  async batch(inks) {
    const { gpu, H } = this, { device } = gpu, P = gpu.readerPipes, n = inks.length, pipes = gpu.tiledPipes('reader', n);
    const feats = inks.map(readerFeats), N = inks.map(i => i.length / 3), Nmax = Math.max(...N);
    const L2 = N.map(v => Math.floor((v + 1) / 2)), L3 = L2.map(v => Math.floor((v + 1) / 2));
    const T1 = Nmax, T2 = Math.floor((Nmax + 1) / 2), T3 = Math.floor((T2 + 1) / 2), Tmax = T3;
    const x0 = new Float32Array(n * Nmax * 5);
    feats.forEach((f, r) => x0.set(f, r * Nmax * 5));
    const oHS = [0, align(2 * n * H)], oCS = 2 * align(2 * n * H);
    const b = { x: [gpu.f32(x0), gpu.zeros(n * T1 * 64), gpu.zeros(n * T2 * 128), gpu.zeros(n * T3 * 128)], lens: gpu.u32([...N, ...N, ...L2, ...L3]),
                rs: gpu.zeros(3 * align(2 * n * H)), seq: [gpu.zeros(n * Tmax * 2 * H), gpu.zeros(n * Tmax * 2 * H)], lp: gpu.zeros(n * Tmax * N_CHARS), dummy: gpu.zeros(4), dummy2: gpu.zeros(4) };
    // uniform slots: conv 0-2, then one per layer and step, then project
    const u = o => [[n, o.Tmax ?? Tmax, o.Tin ?? Tmax, o.Cin ?? 0, o.Cout ?? 0, o.stride ?? 1, H, o.In ?? 0, o.step ?? 0, N_CHARS, o.olen ?? 3 * n, o.olenIn ?? 3 * n,
                     o.oHSp ?? 0, o.oHSn ?? 0, oCS, o.gK ?? 0, o.gKp ?? 0, o.gN ?? 0, o.oW0 ?? 0, o.oB0 ?? 0, o.oW1 ?? 0, o.oB1 ?? 0, 0, 0, 0, 0]];
    const slots = [u({ Tmax: T1, Tin: Nmax, Cin: 5, Cout: 64, stride: 1, olen: n, olenIn: 0, oW0: this.conv[0].oW, oB0: this.conv[0].oB }),
                   u({ Tmax: T2, Tin: T1, Cin: 64, Cout: 128, stride: 2, olen: 2 * n, olenIn: n, oW0: this.conv[1].oW, oB0: this.conv[1].oB }),
                   u({ Tmax: T3, Tin: T2, Cin: 128, Cout: 128, stride: 2, olen: 3 * n, olenIn: 2 * n, oW0: this.conv[2].oW, oB0: this.conv[2].oB })];
    const layerSlot = [];
    for (let l = 0; l < this.layers; l++) {
      const In = l ? 2 * H : 128, [f, bw] = this.lstm[l];
      layerSlot.push(slots.length);
      for (let s = 0; s < Tmax; s++) slots.push(u({ In, step: s, oHSp: oHS[s % 2], oHSn: oHS[(s + 1) % 2], gK: f.K, gKp: f.Kp, gN: f.N, oW0: f.oW, oB0: f.oB, oW1: bw.oW, oB1: bw.oB }));
    }
    const projSlot = slots.length;
    slots.push(u({ In: this.out.In, oW0: this.out.oW, oB0: this.out.oB }));
    const uni = gpu.slots(slots);
    const bind = (xin, xout) => gpu.bindGroup(gpu.readerLayout, uni, [b.lens, xin, xout, b.rs, this.wts, b.dummy, b.dummy2]);   // two unused writable bindings must not alias
    const enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    [[b.x[0], b.x[1], T1], [b.x[1], b.x[2], T2], [b.x[2], b.x[3], T3]].forEach(([xin, xout, T], i) => gpu.run(pass, P.conv, bind(xin, xout), i, n * T * this.conv[i].Cout));
    pass.end();
    let xin = b.x[3];
    for (let l = 0; l < this.layers; l++) {
      const seq = b.seq[l % 2], bg = bind(xin, seq);
      enc.clearBuffer(b.rs);                       // fresh h and c for this layer
      pass = enc.beginComputePass();
      for (let s = 0; s < Tmax; s++) gpu.dispatch(pass, pipes.lstm, bg, layerSlot[l] + s, Math.ceil(H / 64), 1, 2);
      pass.end();
      xin = seq;
    }
    pass = enc.beginComputePass();
    gpu.run(pass, P.project, bind(xin, b.lp), projSlot, n * Tmax);
    pass.end();
    device.queue.submit([enc.finish()]);
    const lp = new Float32Array(await gpu.read(b.lp, n * Tmax * N_CHARS * 4));
    for (const v of Object.values(b)) (Array.isArray(v) ? v : [v]).forEach(x => x.destroy());
    uni.destroy();
    return inks.map((_, r) => ({ lp: lp.slice(r * Tmax * N_CHARS, (r * Tmax + L3[r]) * N_CHARS), T: L3[r] }));
  }
}
