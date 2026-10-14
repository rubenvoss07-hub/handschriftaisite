// Checks picking, layout and the GoodNotes writer against Python (fixtures from web/test/dump.py); writes
// fixtures/out.goodnotes, which compare_gn.py checks against expected.goodnotes. Usage: node web/test/test_layout.mjs
import fs from 'node:fs'; import path from 'node:path';
import { parsePlain } from '../js/bundle.js';
import { SynthCPU } from '../js/synth_cpu.js';
import { ReaderCPU } from '../js/reader_cpu.js';
import { wordSpans, cutAfterWords, cer, bestOf, bestOfChunks, toStrokes } from '../js/pick.js';
import { widen, smoothStroke, letterBaseline, placeLine, BASELINES, BOX, PAGE_W, GN_SIZE, writeText } from '../js/layout.js';
import { writeGoodnotes, readZip, splitStream, strictParse } from '../js/goodnotes.js';

const here = path.dirname(new URL(import.meta.url).pathname), dir = path.join(here, 'fixtures');
const fx = JSON.parse(fs.readFileSync(path.join(dir, 'fixtures.json'), 'utf8'));
let fails = 0;
const check = (name, ok, extra = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${extra}`); };
const flat = a => Float32Array.from(ArrayBuffer.isView(a) ? a : a.flat(2));
const maxDiff = (a, b) => { const A = flat(a), B = flat(b); if (A.length !== B.length) return Infinity; let m = 0; for (let i = 0; i < A.length; i++) m = Math.max(m, Math.abs(A[i] - B[i])); return m; };
const ink = flat(fx.inks[0]), ink1 = flat(fx.inks[1]);

check('baselines match the template', maxDiff(BASELINES, fx.baselines) < 1e-6 && BOX === fx.box && PAGE_W === fx.page_w);
check('word spans', maxDiff(wordSpans(ink, fx.text), fx.word_spans) < 1e-4, JSON.stringify(wordSpans(ink, fx.text).map(s => s.map(v => +v.toFixed(2)))));
const cut = cutAfterWords(ink, fx.text, 2);
check('cut after 2 words', (cut === null ? null : cut.length / 3) === fx.cut_after_2, `${cut && cut.length / 3} vs ${fx.cut_after_2}`);
check('cer', cer('Uber die Brüicke', 'Über die Brücke') === 2 / 15);
const wide = widen(ink, fx.text);
check('widen', maxDiff(wide, fx.widened) < 1e-4, `max diff ${maxDiff(wide, fx.widened).toExponential(2)}`);
const strokes = toStrokes(wide).map(s => smoothStroke(s, 2));
check('smoothed strokes', strokes.length === fx.smoothed.length && maxDiff(strokes, fx.smoothed) < 1e-4);
check('letter baseline', Math.abs(letterBaseline(strokes) - fx.letter_baseline) < 1e-5, `${letterBaseline(strokes)} vs ${fx.letter_baseline}`);
const placed = placeLine(wide, BASELINES[1], BOX, PAGE_W - BOX, GN_SIZE, 2);
check('placed line', maxDiff(placed.strokes, fx.placed.strokes) < 1e-3 && Math.abs(placed.shrink - fx.placed.shrink) < 1e-6, `max diff ${maxDiff(placed.strokes, fx.placed.strokes).toExponential(2)}`);

// the GoodNotes writer, with the same predictable UUIDs as the Python run
let counter = 0;
const fakeUuid = () => { counter++; return counter.toString(16).toUpperCase().padStart(8, '0') + '-0000-4000-8000-' + counter.toString(16).toUpperCase().padStart(12, '0'); };
const template = new Uint8Array(fs.readFileSync(path.join(here, '..', 'assets', 'template.goodnotes')));
const proto = new Uint8Array(fs.readFileSync(path.join(here, '..', 'assets', 'stroke_proto.bin')));
const t0 = Date.now();
const out = await writeGoodnotes(fx.gn_pages, { template, proto, uuid: fakeUuid });
fs.writeFileSync(path.join(dir, 'out.goodnotes'), out);
check('goodnotes written', out.length > 1000, `${out.length} bytes, ${counter} uuids (python used ${fx.gn_uuids_used}), ${Date.now() - t0} ms`);
check('uuid count matches python', counter === fx.gn_uuids_used);
const back = await readZip(out);
check('zip round-trips', back.names.length === 11 && back.names[back.names.length - 1].startsWith('notes/'), back.names.join(' '));
check('template zip reads', (await readZip(template)).names.length === 10);
let strict = true;
try { for (const nm of back.names) if (nm.startsWith('notes/')) for (const m of splitStream(back.entries.get(nm))) strictParse(m); } catch (e) { strict = false; console.log(e); }
check('pages are strict', strict);

// picking end to end on the CPU (slow: two candidates)
if (!process.argv.includes('--quick')) {
  const bundle = parsePlain(new Uint8Array(fs.readFileSync(path.join(dir, 'plain.bin'))));
  const synth = new SynthCPU(bundle.generator), readers = bundle.readers.map(r => new ReaderCPU(r));
  let t = Date.now();
  const b = await bestOf(synth, readers, 'Das ist ein Test.', 2, { seed: 1 });
  check('bestOf picks the lower nll', b.ranked.length === 2 && b.ranked[0].nll <= b.ranked[1].nll && b.ink === b.ranked[0].ink, `${JSON.stringify(b.ranked[0].read)} ${Date.now() - t} ms`);
  t = Date.now();
  const c = await bestOfChunks(synth, readers, 'Das ist ein Test.', 2, { seed: 1, words: 2 });
  check('bestOfChunks runs', c.ink.length / 3 > 100 && typeof c.ranked[0].read === 'string', `${JSON.stringify(c.ranked[0].read)} ${c.ink.length / 3} rows ${Date.now() - t} ms`);
  t = Date.now();
  const lines = [];
  const { pages } = await writeText(synth, readers, 'Hallo Welt.\n\nZweite Zeile hier.', { n: 1, chunkWords: 0, log: (k, line, note) => lines.push(`${k} ${line} ${note}`) });
  check('writeText: two lines on one page, blank rule between', pages.length === 1 && lines.length === 2 && pages[0].length > 4, `${lines.join(' | ')} ${Date.now() - t} ms`);
  const ys = pages[0].flat().map(p => p[1]);
  check('writeText: ink sits on the page', Math.min(...ys) > 0 && Math.max(...ys) < 400);
}
console.log(fails ? `${fails} FAILED` : 'ALL OK');
process.exit(fails ? 1 : 0);
