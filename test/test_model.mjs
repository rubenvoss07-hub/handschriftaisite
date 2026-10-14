// Checks the JavaScript generator and readers against PyTorch (fixtures from web/test/dump.py).
// Usage: node web/test/test_model.mjs
import fs from 'node:fs'; import path from 'node:path';
import { parsePlain } from '../js/bundle.js';
import { encodeText } from '../js/vocab.js';
import { SynthCPU } from '../js/synth_cpu.js';
import { ReaderCPU, readerFeats, greedy, ctcNll } from '../js/reader_cpu.js';

const dir = path.join(path.dirname(new URL(import.meta.url).pathname), 'fixtures');
const fx = JSON.parse(fs.readFileSync(path.join(dir, 'fixtures.json'), 'utf8'));
const bundle = parsePlain(new Uint8Array(fs.readFileSync(path.join(dir, 'plain.bin'))));
let fails = 0;
const check = (name, ok, extra = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${extra}`); };
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const maxDiff2 = (A, B) => Math.max(...A.map((a, i) => maxDiff(a, B[i])));

for (const [t, ids] of Object.entries(fx.encode)) check(`encode ${JSON.stringify(t)}`, encodeText(t).join() === ids.join());

const synth = new SynthCPU(bundle.generator);
const [sx, sy] = fx.stats;
const norm = rows => rows.map(r => Float32Array.from([r[0] / sx, r[1] / sy, r[2]]));
let t0 = Date.now();
let rep = synth.replay(fx.text, norm(fx.raw[0].slice(0, -1)));
console.log(`replay ${rep.y.length} steps in ${Date.now() - t0} ms`);
check('replay: head outputs match torch', maxDiff2(rep.y, fx.replay.y) < 2e-3, `max diff ${maxDiff2(rep.y, fx.replay.y).toExponential(2)}`);
check('replay: window phi matches', maxDiff2(rep.phi, fx.replay.phi) < 2e-3, `max diff ${maxDiff2(rep.phi, fx.replay.phi).toExponential(2)}`);
check('replay: kappa matches', maxDiff2(rep.kappa, fx.replay.kappa) < 2e-3);

const pink = norm(fx.prime_ink);
const first = Float32Array.from(pink[pink.length - 1]); first[2] = 1;
rep = synth.replay(fx.text, [first, ...norm(fx.raw_primed.slice(1, -1))], { primeText: fx.prime_text, primeRows: pink.slice(0, -1) });
check('primed replay: head outputs match torch', maxDiff2(rep.y, fx.replay_primed.y) < 2e-3, `max diff ${maxDiff2(rep.y, fx.replay_primed.y).toExponential(2)}`);
check('primed replay: phi matches', maxDiff2(rep.phi, fx.replay_primed.phi) < 2e-3);

// sampling runs end to end and stops on its own; a seed reproduces its line
t0 = Date.now();
const inks = await synth.sampleN(fx.text, 2, { bias: 1.0, seed: 11 });
console.log(`sampled 2 x ${inks.map(i => i.length / 3).join('+')} rows in ${Date.now() - t0} ms`);
check('sampleN: plausible length', inks.every(i => i.length / 3 > 200 && i.length / 3 < 1500), `${inks.map(i => i.length / 3)}`);
check('sampleN: pen-up at both ends', inks.every(i => i[2] === 1 && i[i.length - 1] === 1));
const again = await synth.sampleN(fx.text, 2, { bias: 1.0, seed: 11 });
check('sampleN: same seed, same ink', again.every((a, i) => a.length === inks[i].length && maxDiff(a, inks[i]) === 0));
const primed = await synth.sampleN(fx.text, 1, { seed: 3, prime: { text: fx.prime_text, ink: Float32Array.from(fx.prime_ink.flat()) } });
check('sampleN with priming runs', primed[0].length / 3 > 100, `${primed[0].length / 3} rows`);

for (const [i, ink] of fx.inks.entries()) {
  const f = readerFeats(Float32Array.from(ink.flat()));
  check(`reader feats ${i} match`, maxDiff(f, fx.reader_feats[i].flat()) < 1e-4);
}
for (const [ri, rfx] of fx.readers.entries()) {
  const rd = new ReaderCPU(bundle.readers[ri]);
  for (const [i, ink] of fx.inks.entries()) {
    t0 = Date.now();
    const { lp, T } = rd.logProbs(Float32Array.from(ink.flat()));
    const want = rfx.logp[i].flat();
    check(`reader ${ri} ink ${i}: frames`, T === rfx.frames[i], `${T} vs ${rfx.frames[i]} (${Date.now() - t0} ms)`);
    check(`reader ${ri} ink ${i}: log-probs match`, maxDiff(lp, want) < 2e-3, `max diff ${maxDiff(lp, want).toExponential(2)}`);
    check(`reader ${ri} ink ${i}: reads ${JSON.stringify(rfx.reads[i])}`, greedy(lp, T) === rfx.reads[i], JSON.stringify(greedy(lp, T)));
    const nll = ctcNll(lp, T, fx.text);
    check(`reader ${ri} ink ${i}: CTC nll`, Math.abs(nll - rfx.nll[i]) < 2e-3, `${nll.toFixed(4)} vs ${rfx.nll[i].toFixed(4)}`);
  }
}
console.log(fails ? `${fails} FAILED` : 'ALL OK');
process.exit(fails ? 1 : 0);
