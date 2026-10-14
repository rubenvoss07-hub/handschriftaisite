// From a line of ink to strokes on the lined page (hw.py widen, letter_baseline, place_line, write_text).
import { toStrokes, wordSpans, strokeWords, bestOf, bestOfChunks, words } from './pick.js';

// The template's rules (vendor template.goodnotes, read by handwrite.paper_baselines) and page geometry
export const BASELINES = [92.125, 124.208, 156.292, 188.375, 220.458, 252.542, 284.625, 316.708, 348.792, 380.875,
  412.958, 445.042, 477.125, 509.208, 541.292, 573.375, 605.458, 637.542, 669.625, 701.708, 733.792, 765.875,
  797.958, 830.042, 862.125, 894.208, 926.292, 958.375, 990.458, 1022.542];
export const PAGE_W = 834.24, PAGE_H = 1078.825, BOX = 29.333333333333332;
export const GN_SIZE = 11.0;          // x-height in page units
export const CHAR_W = 1.54;           // his median ink width per character, in x-heights
export const WIDEN_LETTERS = 1.2, WIDEN_GAPS = 1.3;

export function smoothStroke(s, passes) {
  if (s.length <= 2) return s.map(p => [p[0], p[1]]);
  let a = s.map(p => [p[0], p[1]]);
  for (let k = 0; k < passes; k++) {
    const b = [a[0]];
    for (let i = 1; i < a.length - 1; i++) b.push([0.25 * a[i - 1][0] + 0.5 * a[i][0] + 0.25 * a[i + 1][0], 0.25 * a[i - 1][1] + 0.5 * a[i][1] + 0.25 * a[i + 1][1]]);
    b.push(a[a.length - 1]);
    a = b;
  }
  return a;
}

function percentile(values, p) {
  const s = Float64Array.from(values).sort(), pos = (s.length - 1) * p, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Where the letters stand: the 60th percentile of the strokes' low turning points. */
export function letterBaseline(strokes) {
  const ys = [];
  for (const s of strokes) {
    for (let i = 1; i < s.length - 1; i++) if (s[i][1] >= s[i - 1][1] && s[i][1] > s[i + 1][1]) ys.push(s[i][1]);
    if (s.length > 1 && s[s.length - 1][1] > s[s.length - 2][1]) ys.push(s[s.length - 1][1]);
  }
  if (ys.length) return percentile(ys, 0.6);
  return Math.max(...strokes.flat().map(p => p[1]));
}

/** Stretches every word by `letters` and every gap between words by `gaps` (heights stay). */
export function widen(ink, text, letters = WIDEN_LETTERS, gaps = WIDEN_GAPS) {
  const spans = wordSpans(ink, text);
  if (!spans.length || (letters === 1 && gaps === 1)) return ink;
  const starts = [spans[0][0]];
  for (let i = 1; i < spans.length; i++) starts.push(starts[i - 1] + (spans[i - 1][1] - spans[i - 1][0]) * letters + (spans[i][0] - spans[i - 1][1]) * gaps);
  const { sid, xs, strokes } = strokeWords(ink, spans), N = ink.length / 3, nx = new Float64Array(N);
  const wordOf = new Int32Array(strokes.length ? Math.max(...strokes.map(s => s.sid)) + 1 : 0);
  for (const s of strokes) wordOf[s.sid] = s.word;
  for (let t = 0; t < N; t++) { const w = wordOf[sid[t]]; nx[t] = starts[w] + (xs[t] - spans[w][0]) * letters; }
  const out = Float32Array.from(ink);
  let prev = 0;
  for (let t = 0; t < N; t++) { out[3 * t] = nx[t] - prev; prev = nx[t]; }
  return out;
}

/** Ink offsets -> strokes in page units with the letters standing on the rule; also the shrink factor. */
export function placeLine(ink, baseline, left, right, size, passes) {
  const strokes = toStrokes(ink).map(s => smoothStroke(s, passes));
  const pts = strokes.flat();
  const x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0])), base = letterBaseline(strokes);
  let k = size;
  const width = (x1 - x0) * k;
  if (width > right - left) k *= (right - left) / width;
  return { strokes: strokes.map(s => s.map(([x, y]) => [left + (x - x0) * k, baseline + (y - base) * k])), shrink: k / size };
}

/** Text -> pages of GoodNotes strokes, each line the best of n (hw.py write_text). log(line, note) per line;
 *  shouldStop() lets the page cancel between lines. */
export async function writeText(synth, readers, text, { n = 64, size = GN_SIZE, smooth = 2, chunkWords = 2,
    widenLetters = WIDEN_LETTERS, widenGaps = WIDEN_GAPS, seed = 0, log = () => {}, shouldStop = () => false } = {}) {
  const left = BOX, right = PAGE_W - BOX, avail = right - left;
  let est = CHAR_W * size, pages = [], cur = [], row = 0, nLine = 0;
  for (const para of text.split('\n')) {
    const ws = words(para);
    let i = 0;
    if (!ws.length) row++;
    while (i < ws.length) {
      if (shouldStop()) throw new Error('abgebrochen');
      if (row >= BASELINES.length) { pages.push(cur); cur = []; row = 0; }
      let j = i + 1;
      while (j < ws.length && ws.slice(i, j + 1).join(' ').length * est <= avail) j++;
      let ink, ranked, line;
      for (let tries = 0; tries < 3; tries++) {
        line = ws.slice(i, j).join(' ');
        const r = chunkWords && n > 1 ? await bestOfChunks(synth, readers, line, n, { seed: seed + 97 * nLine, words: chunkWords })
                                      : await bestOf(synth, readers, line, n, { seed: seed + 97 * nLine });
        ink = widen(r.ink, line, widenLetters, widenGaps); ranked = r.ranked;
        let x = 0, lo = 0, hi = 0;
        for (let t = 0; t < ink.length / 3; t++) { x += ink[3 * t]; lo = Math.min(lo, x); hi = Math.max(hi, x); }
        const width = (hi - lo) * size;
        est = 0.6 * est + 0.4 * width / line.length;
        if (width <= avail * 1.02 || j - i === 1) break;
        j--;
      }
      const { strokes, shrink } = placeLine(ink, BASELINES[row], left, right, size, smooth);
      cur.push(...strokes);
      nLine++;
      const notes = [];
      if (ranked[0].unread.length) notes.push('unklar: ' + ranked[0].unread.join(', '));
      if (shrink < 1) notes.push(`auf ${Math.round(shrink * 100)}% verkleinert`);
      log(nLine, line, notes.join('; '));
      row++;
      i = j;
    }
  }
  pages.push(cur);
  return { pages, baselines: BASELINES };
}
