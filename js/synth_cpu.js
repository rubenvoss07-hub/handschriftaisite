// The Graves generator (hw.py build_model) on the CPU, in plain JavaScript. Slow (good for one or two
// candidates, and as the reference the WebGPU engine is checked against); the shapes match hw.py exactly.
import { encodeText, N_CHARS } from './vocab.js';
import { uniform, normal2 } from './rng.js';

const sigmoid = x => 1 / (1 + Math.exp(-x));

/** y[j] = b[j] + sum_k W[j, k] * x[k] for a row-major (J, K) matrix; x may be a concatenation (parts). */
function gemv(W, J, K, parts, y, b) {
  for (let j = 0; j < J; j++) {
    let acc = b ? b[j] : 0, base = j * K, k = 0;
    for (const p of parts) {
      for (let i = 0; i < p.length; i++) acc += W[base + k + i] * p[i];
      k += p.length;
    }
    y[j] = acc;
  }
}

/** One LSTM cell update in place: gates (4H) in PyTorch order i, f, g, o. */
function lstmCell(gates, H, h, c) {
  for (let u = 0; u < H; u++) {
    const i = sigmoid(gates[u]), f = sigmoid(gates[H + u]), g = Math.tanh(gates[2 * H + u]), o = sigmoid(gates[3 * H + u]);
    c[u] = f * c[u] + i * g;
    h[u] = o * Math.tanh(c[u]);
  }
}

function cat(name, g) {   // [W_ih | W_hh] side by side and b_ih + b_hh, so one gemv does a layer
  const ih = g.get(name + '.weight_ih' + (name === 'lstm1' ? '' : '_l0')), hh = g.get(name + '.weight_hh' + (name === 'lstm1' ? '' : '_l0'));
  const bi = g.get(name + '.bias_ih' + (name === 'lstm1' ? '' : '_l0')).data, bh = g.get(name + '.bias_hh' + (name === 'lstm1' ? '' : '_l0')).data;
  const [J, Ki] = ih.shape, Kh = hh.shape[1], K = Ki + Kh;
  const W = new Float32Array(J * K), b = new Float32Array(J);
  for (let j = 0; j < J; j++) {
    W.set(ih.data.subarray(j * Ki, (j + 1) * Ki), j * K);
    W.set(hh.data.subarray(j * Kh, (j + 1) * Kh), j * K + Ki);
    b[j] = bi[j] + bh[j];
  }
  return { W, J, K, b };
}

export class SynthCPU {
  constructor(gen) {
    this.H = gen.hidden; this.K = gen.K; this.M = gen.M; this.S = gen.style_dim;
    this.stats = gen.stats; this.writer = gen.writer;
    this.l1 = cat('lstm1', gen); this.l2 = cat('lstm2', gen); this.l3 = cat('lstm3', gen);
    const win = gen.get('window.weight'), head = gen.get('head.weight');
    this.win = { W: win.data, J: win.shape[0], K: win.shape[1], b: gen.get('window.bias').data };
    this.head = { W: head.data, J: head.shape[0], K: head.shape[1], b: gen.get('head.bias').data };
    const st = gen.get('style.weight');
    this.style = this.S ? st.data.subarray(this.writer * this.S, (this.writer + 1) * this.S) : new Float32Array(0);
  }

  /** Fresh recurrent state for n rows of the same text. */
  init(text, n) {
    const ids = encodeText(text), U = ids.length, H = this.H;
    const rows = [];
    for (let r = 0; r < n; r++) {
      rows.push({ h1: new Float32Array(H), c1: new Float32Array(H), h2: new Float32Array(H), c2: new Float32Array(H),
                  h3: new Float32Array(H), c3: new Float32Array(H), w: new Float32Array(N_CHARS),
                  kappa: new Float32Array(this.K), phi: new Float32Array(U + 1), y: new Float32Array(this.head.J) });
    }
    return { ids, U, rows, g1: new Float32Array(this.l1.J), g2: new Float32Array(this.l2.J), g3: new Float32Array(this.l3.J),
             abk: new Float32Array(this.win.J) };
  }

  /** One step for row r with normalized input x (3): updates the state, leaves the head output in row.y. */
  step(st, r, x) {
    const row = st.rows[r], H = this.H, K = this.K, U = st.U, s = this.style;
    gemv(this.l1.W, this.l1.J, this.l1.K, [x, s, row.w, row.h1], st.g1, this.l1.b);
    lstmCell(st.g1, H, row.h1, row.c1);
    gemv(this.win.W, this.win.J, this.win.K, [row.h1], st.abk, this.win.b);
    const phi = row.phi, kappa = row.kappa;
    for (let j = 0; j < K; j++) kappa[j] += Math.exp(st.abk[2 * K + j]) / 25;
    for (let u = 0; u <= U; u++) {
      let p = 0;
      for (let j = 0; j < K; j++) {
        const d = kappa[j] - u;
        p += Math.exp(st.abk[j]) * Math.exp(-Math.exp(st.abk[K + j]) * d * d);
      }
      phi[u] = p;
    }
    row.w.fill(0);
    for (let u = 0; u < U; u++) if (st.ids[u]) row.w[st.ids[u]] += phi[u];
    gemv(this.l2.W, this.l2.J, this.l2.K, [x, s, row.h1, row.w, row.h2], st.g2, this.l2.b);
    lstmCell(st.g2, H, row.h2, row.c2);
    gemv(this.l3.W, this.l3.J, this.l3.K, [x, s, row.h2, row.w, row.h3], st.g3, this.l3.b);
    lstmCell(st.g3, H, row.h3, row.c3);
    gemv(this.head.W, this.head.J, this.head.K, [row.h1, row.h2, row.h3], row.y, this.head.b);
  }

  /** Draws (dx, dy, pen) from the head output y at this bias with the row's random slots. */
  draw(y, bias, seed, step, r) {
    const M = this.M, t = 1 + bias;
    let mx = -Infinity;
    for (let k = 0; k < M; k++) mx = Math.max(mx, y[1 + k] * t);
    let z = 0;
    for (let k = 0; k < M; k++) z += Math.exp(y[1 + k] * t - mx);
    const u = uniform(seed, step, r, 0);
    let k = 0, acc = 0;
    for (; k < M - 1; k++) {
      acc += Math.exp(y[1 + k] * t - mx) / z;
      if (u < acc) break;
    }
    const [z1, z2] = normal2(seed, step, r, 1);
    const sx = Math.exp(y[1 + 3 * M + k] - bias), sy = Math.exp(y[1 + 4 * M + k] - bias), rho = Math.tanh(y[1 + 5 * M + k]);
    const dx = y[1 + M + k] + sx * z1, dy = y[1 + 2 * M + k] + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
    const pen = uniform(seed, step, r, 3) < sigmoid(y[0]) ? 1 : 0;
    return [dx, dy, pen];
  }

  /** n versions of text at once; prime = {text, ink} first feeds a written line (Graves priming).
   *  Returns Float32Array inks (rows of dx, dy, pen in x-height units; the first and last row pen-up). */
  async sampleN(text, n, { bias = 1.0, seed = 0, prime = null, maxSteps = 0, endFrac = 0.5 } = {}) {
    const t0 = Date.now();
    const [sx, sy] = this.stats;
    const st = this.init(prime ? prime.text + ' ' + text : text, n);
    const U = st.U, x = new Float32Array(3);
    let x0 = [0, 0, 0];
    if (prime) {
      const p = prime.ink, T = p.length / 3;
      for (let t = 0; t < T - 1; t++) {
        x[0] = p[3 * t] / sx; x[1] = p[3 * t + 1] / sy; x[2] = p[3 * t + 2];
        for (let r = 0; r < n; r++) this.step(st, r, x);
      }
      x0 = [p[3 * (T - 1)] / sx, p[3 * (T - 1) + 1] / sy, 1];
    }
    const outs = [], cur = [], done = new Uint8Array(n);
    for (let r = 0; r < n; r++) { outs.push([0, 0, 0]); cur.push(Float32Array.from(x0)); }
    const limit = maxSteps || 40 * U + 100;
    let left = n;
    let steps = 0;
    for (let step = 0; step < limit && left > 0; step++) {
      if (step % 8 === 7) await new Promise(r => setTimeout(r, 0));   // let the page breathe
      steps++;
      for (let r = 0; r < n; r++) {
        if (done[r]) continue;
        const row = st.rows[r];
        this.step(st, r, cur[r]);
        const [dx, dy, pen] = this.draw(row.y, bias, seed, step, r);
        outs[r].push(dx * sx, dy * sy, pen);
        let peak = 0;
        for (let u = 0; u < U; u++) peak = Math.max(peak, row.phi[u]);
        if (row.phi[U] > endFrac * peak && pen > 0) { done[r] = 1; left--; }
        cur[r][0] = dx; cur[r][1] = dy; cur[r][2] = pen;
      }
    }
    this.lastRun = { steps, ms: Date.now() - t0, n };
    return outs.map(o => {
      const ink = Float32Array.from(o);
      ink[2] = 1; ink[ink.length - 1] = 1;
      return ink;
    });
  }

  /** For tests: feeds given normalized rows (after optional priming rows) and returns y, phi, kappa per step. */
  replay(text, inputs, { primeText = null, primeRows = null } = {}) {
    const st = this.init(primeText ? primeText + ' ' + text : text, 1), row = st.rows[0];
    if (primeRows) for (const x of primeRows) this.step(st, 0, x);
    const out = { y: [], phi: [], kappa: [] };
    for (const x of inputs) {
      this.step(st, 0, x);
      out.y.push(Array.from(row.y)); out.phi.push(Array.from(row.phi)); out.kappa.push(Array.from(row.kappa));
    }
    return out;
  }
}
