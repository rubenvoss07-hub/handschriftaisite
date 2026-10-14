// The local readers (hw.py build_reader: conv + BiLSTM + CTC) on the CPU: transcribe an ink and score how
// clearly it writes the intended text (CTC negative log-likelihood per character, capped at 10).
import { readerIds, idToChar, N_CHARS } from './vocab.js';

const READER_SCALE = 5.0;
const sigmoid = x => 1 / (1 + Math.exp(-x));

function percentile(sorted, p) {      // numpy's default linear interpolation
  const pos = (sorted.length - 1) * p, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** (N, 3) ink -> (N, 5) features, as hw.py reader_feats. */
export function readerFeats(ink) {
  const N = ink.length / 3, y = new Float64Array(N);
  let acc = 0;
  for (let t = 0; t < N; t++) { acc += ink[3 * t + 1]; y[t] = acc; }
  const sorted = Float64Array.from(y).sort();
  const q25 = percentile(sorted, 0.25), q75 = percentile(sorted, 0.75), med = percentile(sorted, 0.5);
  const k = Math.min(2.5, Math.max(0.4, 0.7 / Math.max(q75 - q25, 1e-3)));
  const f = new Float32Array(N * 5);
  const clip = v => Math.min(3, Math.max(-3, v));
  for (let t = 0; t < N; t++) {
    f[5 * t] = clip(ink[3 * t] * k) * READER_SCALE;
    f[5 * t + 1] = clip(ink[3 * t + 1] * k) * READER_SCALE;
    f[5 * t + 2] = ink[3 * t + 2];
    f[5 * t + 3] = t ? ink[3 * (t - 1) + 2] : 0;
    f[5 * t + 4] = (y[t] - med) * k;
  }
  return f;
}

function conv1d(x, T, Cin, w, b, Cout, stride) {   // kernel 5, padding 2, ReLU; x is (T, Cin) row-major
  const To = Math.floor((T + 4 - 5) / stride) + 1, out = new Float32Array(To * Cout);
  for (let t = 0; t < To; t++) {
    for (let o = 0; o < Cout; o++) {
      let acc = b[o];
      for (let j = 0; j < 5; j++) {
        const ti = t * stride + j - 2;
        if (ti < 0 || ti >= T) continue;
        const xb = ti * Cin, wb = (o * Cin) * 5 + j;
        for (let i = 0; i < Cin; i++) acc += w[wb + i * 5] * x[xb + i];
      }
      out[t * Cout + o] = acc > 0 ? acc : 0;
    }
  }
  return [out, To];
}

function lstmDir(x, T, In, W, H, reverse) {   // one direction over (T, In); returns (T, H)
  const { ih, hh, b } = W, out = new Float32Array(T * H), h = new Float32Array(H), c = new Float32Array(H), g = new Float32Array(4 * H);
  for (let s = 0; s < T; s++) {
    const t = reverse ? T - 1 - s : s, xb = t * In;
    for (let j = 0; j < 4 * H; j++) {
      let acc = b[j];
      const ib = j * In, hb = j * H;
      for (let i = 0; i < In; i++) acc += ih[ib + i] * x[xb + i];
      for (let i = 0; i < H; i++) acc += hh[hb + i] * h[i];
      g[j] = acc;
    }
    for (let u = 0; u < H; u++) {
      const i = sigmoid(g[u]), f = sigmoid(g[H + u]), gg = Math.tanh(g[2 * H + u]), o = sigmoid(g[3 * H + u]);
      c[u] = f * c[u] + i * gg;
      h[u] = o * Math.tanh(c[u]);
    }
    out.set(h, t * H);
  }
  return out;
}

export class ReaderCPU {
  constructor(rd) {
    this.H = rd.cfg.hidden; this.layers = rd.cfg.layers;
    this.conv = [0, 2, 4].map(i => ({ w: rd.get(`conv.${i}.weight`), b: rd.get(`conv.${i}.bias`).data }));
    this.lstm = [];
    for (let l = 0; l < this.layers; l++) {
      this.lstm.push(['', '_reverse'].map(sfx => {
        const ih = rd.get(`lstm.weight_ih_l${l}${sfx}`), bi = rd.get(`lstm.bias_ih_l${l}${sfx}`).data, bh = rd.get(`lstm.bias_hh_l${l}${sfx}`).data;
        const b = new Float32Array(bi.length);
        for (let j = 0; j < b.length; j++) b[j] = bi[j] + bh[j];
        return { ih: ih.data, In: ih.shape[1], hh: rd.get(`lstm.weight_hh_l${l}${sfx}`).data, b };
      }));
    }
    this.out = { w: rd.get('out.weight').data, b: rd.get('out.bias').data };
  }

  /** The engine interface pick.js uses: [{lp, T}] per ink, yielding to the page between inks. */
  async logProbsBatch(inks) {
    const out = [];
    for (const ink of inks) { out.push(this.logProbs(ink)); await new Promise(r => setTimeout(r, 0)); }
    return out;
  }

  /** Log-probabilities (T', N_CHARS) for one ink. */
  logProbs(ink) {
    let x = readerFeats(ink), T = ink.length / 3, C = 5;
    for (const [i, cv] of this.conv.entries()) {
      [x, T] = conv1d(x, T, C, cv.w.data, cv.b, cv.w.shape[0], i ? 2 : 1);
      C = cv.w.shape[0];
    }
    const H = this.H;
    for (const [fw, bw] of this.lstm) {
      const a = lstmDir(x, T, C, fw, H, false), b = lstmDir(x, T, C, bw, H, true), y = new Float32Array(T * 2 * H);
      for (let t = 0; t < T; t++) { y.set(a.subarray(t * H, (t + 1) * H), t * 2 * H); y.set(b.subarray(t * H, (t + 1) * H), t * 2 * H + H); }
      x = y; C = 2 * H;
    }
    const lp = new Float32Array(T * N_CHARS);
    for (let t = 0; t < T; t++) {
      let mx = -Infinity;
      for (let o = 0; o < N_CHARS; o++) {
        let acc = this.out.b[o];
        const wb = o * C, xb = t * C;
        for (let i = 0; i < C; i++) acc += this.out.w[wb + i] * x[xb + i];
        lp[t * N_CHARS + o] = acc;
        if (acc > mx) mx = acc;
      }
      let z = 0;
      for (let o = 0; o < N_CHARS; o++) z += Math.exp(lp[t * N_CHARS + o] - mx);
      const lz = mx + Math.log(z);
      for (let o = 0; o < N_CHARS; o++) lp[t * N_CHARS + o] -= lz;
    }
    return { lp, T };
  }
}

/** Greedy CTC transcription of log-probs (T, N_CHARS). */
export function greedy(lp, T) {
  let prev = 0, out = '';
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -Infinity;
    for (let o = 0; o < N_CHARS; o++) if (lp[t * N_CHARS + o] > bv) { bv = lp[t * N_CHARS + o]; best = o; }
    if (best !== prev && best >= 2) out += idToChar(best);
    prev = best;
  }
  return out;
}

function logAdd(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}

/** CTC negative log-likelihood of text per character, capped at 10 (hw.py reader_run). */
export function ctcNll(lp, T, text) {
  const ids = readerIds(text), L = ids.length ? ids : [1], S = 2 * L.length + 1, ext = new Int32Array(S);
  for (let i = 0; i < L.length; i++) ext[2 * i + 1] = L[i];
  let alpha = new Float64Array(S).fill(-Infinity);
  alpha[0] = lp[0]; if (S > 1) alpha[1] = lp[ext[1]];
  for (let t = 1; t < T; t++) {
    const next = new Float64Array(S).fill(-Infinity), base = t * N_CHARS;
    for (let s = 0; s < S; s++) {
      let a = alpha[s];
      if (s > 0) a = logAdd(a, alpha[s - 1]);
      if (s > 1 && ext[s] !== 0 && ext[s] !== ext[s - 2]) a = logAdd(a, alpha[s - 2]);
      next[s] = a === -Infinity ? -Infinity : a + lp[base + ext[s]];
    }
    alpha = next;
  }
  const ll = S > 1 ? logAdd(alpha[S - 1], alpha[S - 2]) : alpha[0];
  const nll = -ll / L.length;
  return Number.isFinite(nll) ? Math.min(10, nll) : 10;
}
