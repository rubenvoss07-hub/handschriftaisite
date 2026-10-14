// Best-of-n with the readers (hw.py best_of / best_of_chunks): the generator writes n versions, the readers
// keep the one that most clearly says the intended text. Engines are async so the WebGPU ones fit too:
// synth.sampleN(text, n, opts) -> inks; reader.logProbsBatch(inks) -> [{lp, T}].
import { greedy, ctcNll } from './reader_cpu.js';

/** (N, 3) offsets -> strokes, each an array of [x, y] points (hw.py to_strokes). */
export function toStrokes(ink) {
  const strokes = [];
  let cur = [], x = 0, y = 0;
  for (let t = 0; t < ink.length / 3; t++) {
    x += ink[3 * t]; y += ink[3 * t + 1];
    cur.push([x, y]);
    if (ink[3 * t + 2] > 0.5) { strokes.push(cur); cur = []; }
  }
  if (cur.length) strokes.push(cur);
  return strokes;
}

/** [[left, right], ...] of the words in a line of ink (hw.py word_spans). */
export function wordSpans(ink, text, minGap = 1.0) {
  const iv = toStrokes(ink).filter(s => s.length > 1)
    .map(s => [Math.min(...s.map(p => p[0])), Math.max(...s.map(p => p[0]))])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (!iv.length) return [];
  const merged = [[...iv[0]]];
  for (const [lo, hi] of iv.slice(1)) {
    if (lo <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], hi);
    else merged.push([lo, hi]);
  }
  const nWords = words(text).length;
  const holes = [...Array(merged.length - 1).keys()].sort((i, j) => (merged[i][1] - merged[i + 1][0]) - (merged[j][1] - merged[j + 1][0]) || i - j);
  const cuts = holes.slice(0, nWords - 1).filter(i => merged[i + 1][0] - merged[i][1] >= minGap).sort((a, b) => a - b);
  const spans = [];
  let start = 0;
  for (const i of [...cuts, merged.length - 1]) { spans.push([merged[start][0], merged[i][1]]); start = i + 1; }
  return spans;
}

export const words = text => text.split(/\s+/).filter(Boolean);

/** Which word a stroke belongs to: the nearest span to its horizontal centre (first on ties). */
export function strokeWords(ink, spans) {
  const N = ink.length / 3, sid = new Int32Array(N), out = [];
  let x = 0, s = 0;
  const xs = new Float64Array(N);
  for (let t = 0; t < N; t++) { x += ink[3 * t]; xs[t] = x; sid[t] = s; if (ink[3 * t + 2] > 0.5) s++; }
  const nS = N ? sid[N - 1] + 1 : 0;
  for (let k = 0; k < nS; k++) {
    let lo = Infinity, hi = -Infinity, first = -1;
    for (let t = 0; t < N; t++) if (sid[t] === k) { lo = Math.min(lo, xs[t]); hi = Math.max(hi, xs[t]); if (first < 0) first = t; }
    if (first < 0) continue;   // a pen-up row right at the end leaves an empty stroke id
    const c = (lo + hi) / 2;
    let best = 0, bd = Infinity;
    spans.forEach(([l, r], w) => { const d = l <= c && c <= r ? 0 : Math.min(Math.abs(c - l), Math.abs(c - r)); if (d < bd) { bd = d; best = w; } });
    out.push({ sid: k, first, word: best, lo });
  }
  return { sid, xs, strokes: out };
}

/** The rows of a line up to the end of its first k words, or null (hw.py cut_after_words). */
export function cutAfterWords(ink, text, k) {
  const spans = wordSpans(ink, text);
  if (spans.length !== words(text).length) return null;
  for (const s of strokeWords(ink, spans).strokes) {
    if (s.word >= k) return s.first > 1 ? ink.slice(0, 3 * s.first) : null;
  }
  return null;
}

/** Character error rate of got against want. */
export function cer(got, want) {
  let prev = [...Array(want.length + 1).keys()];
  const g = [...got], w = [...want];
  for (let i = 1; i <= g.length; i++) {
    const cur = [i];
    for (let j = 1; j <= w.length; j++) cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (g[i - 1] !== w[j - 1] ? 1 : 0)));
    prev = cur;
  }
  return prev[w.length] / Math.max(1, w.length);
}

async function scores(readers, inks, text) {   // mean CTC nll over the readers, and reader 0's transcriptions
  const nll = new Float64Array(inks.length);
  let reads = null;
  for (const [ri, rd] of readers.entries()) {
    const outs = await rd.logProbsBatch(inks);
    outs.forEach(({ lp, T }, i) => { nll[i] += ctcNll(lp, T, text) / readers.length; });
    if (ri === 0) reads = outs.map(({ lp, T }) => greedy(lp, T));
  }
  return { nll, reads };
}

const argmin = a => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[b]) b = i; return b; };
const unread = (read, text) => { const r = words(read); return words(text).filter(w => !r.includes(w)); };

/** Writes text n times, keeps the clearest. Returns {ink, ranked: [{unread, cer, read, ink}], best first}. */
export async function bestOf(synth, readers, text, n, { seed = 0, bias = 1.0 } = {}) {
  const inks = await synth.sampleN(text, n, { bias, seed });
  if (n <= 1) return { ink: inks[0], ranked: [{ unread: [], cer: 0, read: '', ink: inks[0] }] };
  const { nll, reads } = await scores(readers, inks, text);
  const rows = inks.map((ink, i) => ({ unread: unread(reads[i], text), cer: cer(reads[i], text), read: reads[i], ink, nll: nll[i] }));
  const order = [...rows.keys()].sort((a, b) => nll[a] - nll[b] || a - b);
  return { ink: rows[argmin(nll)].ink, ranked: order.map(i => rows[i]) };
}

function concat(a, b) {   // a + b without b's first (start) row
  const out = new Float32Array(a.length + b.length - 3);
  out.set(a); out.set(b.subarray(3), a.length);
  return out;
}

/** bestOf a few words at a time: the readers pick the best next piece, the model continues from it. */
export async function bestOfChunks(synth, readers, text, n, { seed = 0, bias = 1.0, words: k = 2 } = {}) {
  const ws = words(text);
  let done = 0, ink = null;
  while (done < ws.length) {
    const rest = ws.slice(done).join(' ');
    const prime = done ? { text: ws.slice(0, done).join(' '), ink } : null;
    const cands = await synth.sampleN(rest, n, { bias, seed: seed + 7919 * done, prime });
    let upto = Math.min(ws.length, done + k);
    let pieces = upto === ws.length ? cands : cands.map(c => cutAfterWords(c, rest, k));
    if (pieces.every(p => p === null)) { pieces = cands; upto = ws.length; }
    const full = pieces.map(p => p === null ? null : ink === null ? p : concat(ink, p));
    const ok = [...full.keys()].filter(i => full[i] !== null);
    const { nll } = await scores(readers, ok.map(i => full[i]), ws.slice(0, upto).join(' '));
    ink = full[ok[argmin(nll)]];
    done = upto;
  }
  const [{ lp, T }] = await readers[0].logProbsBatch([ink]);
  const read = greedy(lp, T);
  return { ink, ranked: [{ unread: unread(read, text), cer: cer(read, text), read, ink }] };
}
