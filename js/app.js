// The page: unlock the model with the password, write text in Ruben's hand on the iPad, hand the
// .goodnotes file to GoodNotes. Everything runs in this tab; the site only serves files.
import { unlock } from './bundle.js';
import { SynthCPU } from './synth_cpu.js';
import { ReaderCPU } from './reader_cpu.js';
import { initGPU, SynthGPU, ReaderGPU } from './gpu.js';
import { writeText, PAGE_W, PAGE_H } from './layout.js';
import { writeGoodnotes } from './goodnotes.js';

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') || 'model.enc';
const PRESETS = { instant: { n: 1, chunkWords: 0 }, fast: { n: 8, chunkWords: 0 }, good: { n: 64, chunkWords: 0 }, best: { n: 64, chunkWords: 2 } };
let gpu = null, synth = null, readers = null, template = null, proto = null, stopFlag = false, wakeLock = null;

const fmtMB = b => (b / 1e6).toFixed(1) + ' MB';

/** The encrypted bundle: from the Cache API when the server still has the same file, else downloaded. */
async function loadBundle(onProgress) {
  let tag = '';
  try { const h = await fetch(MODEL_URL, { method: 'HEAD', cache: 'no-store' }); tag = (h.headers.get('etag') || '') + '|' + (h.headers.get('content-length') || ''); } catch { }
  const key = MODEL_URL + '#' + tag;
  let cache = null;
  try { cache = await caches.open('handschrift-model'); } catch { }
  if (cache) {
    const hit = await cache.match(key);
    if (hit) { onProgress(1, 'aus dem Speicher'); return new Uint8Array(await hit.arrayBuffer()); }
  }
  const r = await fetch(MODEL_URL, { cache: 'no-store' });
  if (!r.ok) throw new Error(`Modell nicht gefunden (${r.status})`);
  const total = +r.headers.get('content-length') || 0, chunks = [];
  let got = 0;
  const reader = r.body.getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress(total ? got / total : 0, `${fmtMB(got)}${total ? ' von ' + fmtMB(total) : ''}`);
  }
  const bytes = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  if (cache) { try { for (const k of await cache.keys()) await cache.delete(k); await cache.put(key, new Response(bytes)); } catch { } }
  return bytes;
}

async function init() {
  gpu = params.get('engine') === 'cpu' ? null : await initGPU().catch(e => { console.warn(e); return null; });
  $('engine').textContent = gpu ? `Rechnet auf der GPU (WebGPU${gpu.info && gpu.info !== 'WebGPU' ? ': ' + gpu.info : ''}).`
                                : 'Kein WebGPU in diesem Browser: es geht, aber sehr langsam (iPadOS 26 / Safari 26 hat WebGPU).';
  [template, proto] = await Promise.all(['assets/template.goodnotes', 'assets/stroke_proto.bin'].map(async u => new Uint8Array(await (await fetch(u)).arrayBuffer())));
  $('pw').focus();
}

$('loginForm').addEventListener('submit', async e => {
  e.preventDefault();
  const pw = $('pw').value;
  if (!pw) return;
  $('unlock').disabled = true;
  $('loginStatus').classList.remove('error');
  try {
    $('dlProgress').hidden = false;
    const bytes = await loadBundle((f, msg) => { $('dlProgress').value = Math.round(f * 100); $('loginStatus').textContent = 'Modell laden… ' + msg; });
    $('loginStatus').textContent = 'Entschlüsseln…';
    await new Promise(r => setTimeout(r, 30));
    const bundle = await unlock(bytes, pw);
    $('loginStatus').textContent = 'Modell aufbauen…';
    await new Promise(r => setTimeout(r, 30));
    if (gpu) { synth = new SynthGPU(gpu, bundle.generator); readers = bundle.readers.map(r => new ReaderGPU(gpu, r)); }
    else { synth = new SynthCPU(bundle.generator); readers = bundle.readers.map(r => new ReaderCPU(r)); }
    $('login').hidden = true;
    $('write').hidden = false;
    $('text').focus();
  } catch (err) {
    $('loginStatus').textContent = err.message === 'wrong password' ? 'Falsches Passwort.' : 'Fehler: ' + err.message;
    $('loginStatus').classList.add('error');
    $('dlProgress').hidden = true;
  }
  $('unlock').disabled = false;
});

function drawPages(pages, baselines) {
  const box = $('preview');
  box.innerHTML = '';
  const dpr = Math.min(2, devicePixelRatio || 1), scale = Math.min(1.2, (box.clientWidth || 700) / PAGE_W) * dpr;
  for (const strokes of pages) {
    const c = document.createElement('canvas');
    c.width = Math.round(PAGE_W * scale); c.height = Math.round(PAGE_H * scale);
    const g = c.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height);
    g.strokeStyle = '#d0d2d3'; g.lineWidth = 1;
    for (const b of baselines) { g.beginPath(); g.moveTo(0, b * scale); g.lineTo(c.width, b * scale); g.stroke(); }
    g.strokeStyle = '#E67300'; g.lineWidth = 2.25 * scale; g.lineCap = 'round'; g.lineJoin = 'round';
    for (const s of strokes) {
      g.beginPath();
      if (s.length === 1) { g.arc(s[0][0] * scale, s[0][1] * scale, 1.2 * scale, 0, 7); g.fillStyle = '#E67300'; g.fill(); continue; }
      g.moveTo(s[0][0] * scale, s[0][1] * scale);
      for (let i = 1; i < s.length; i++) g.lineTo(s[i][0] * scale, s[i][1] * scale);
      g.stroke();
    }
    box.appendChild(c);
  }
}

let currentBlob = null;
$('go').addEventListener('click', async () => {
  const text = $('text').value.trim();
  if (!text) { $('text').focus(); return; }
  const preset = PRESETS[$('quality').value], seed = +$('seed').value || 0;
  const n = gpu ? preset.n : Math.min(preset.n, 2);
  stopFlag = false;
  $('go').disabled = true; $('stop').hidden = false; $('result').hidden = true; $('lines').innerHTML = ''; $('status').classList.remove('error');
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { }
  const t0 = performance.now();
  const tick = setInterval(() => {
    const s = Math.round((performance.now() - t0) / 1000);
    const speed = synth.lastRun ? ` · Modell ${(synth.lastRun.ms / synth.lastRun.steps).toFixed(1)} ms je Schritt` : '';
    const rd = readers[0].lastRun ? ` · Leser ${(readers[0].lastRun.ms / 1000).toFixed(1)} s je ${readers[0].lastRun.n}` : '';
    $('status').textContent = `Schreibt… ${s} s${speed}${rd}`;
  }, 500);
  try {
    const { pages, baselines } = await writeText(synth, readers, text, { n, chunkWords: preset.chunkWords, seed, shouldStop: () => stopFlag,
      log: (k, line, note) => { const li = document.createElement('li'); li.textContent = line; if (note) { const s = document.createElement('span'); s.className = 'note'; s.textContent = '  (' + note + ')'; li.appendChild(s); } $('lines').appendChild(li); } });
    drawPages(pages, baselines);
    const bytes = await writeGoodnotes(pages, { template, proto });
    currentBlob = new Blob([bytes], { type: 'application/octet-stream' });
    if ($('download').href.startsWith('blob:')) URL.revokeObjectURL($('download').href);
    $('download').href = URL.createObjectURL(currentBlob);
    const file = new File([currentBlob], 'handschrift.goodnotes', { type: 'application/octet-stream' });
    $('share').hidden = !(navigator.canShare && navigator.canShare({ files: [file] }));
    $('result').hidden = false;
    const secs = Math.round((performance.now() - t0) / 1000);
    const speed = synth.lastRun ? ` Modell ${(synth.lastRun.ms / synth.lastRun.steps).toFixed(1)} ms je Schritt` : '';
    const rd = readers[0].lastRun ? `, Leser ${(readers[0].lastRun.ms / 1000).toFixed(1)} s je ${readers[0].lastRun.n}` : '';
    $('status').textContent = `Fertig: ${pages.length} Seite${pages.length > 1 ? 'n' : ''}, ${$('lines').children.length} Zeilen in ${secs} s.${speed}${rd}.`;
    window.__lastResult = { pages: pages.length, lines: $('lines').children.length, bytes: bytes.length, secs };
  } catch (err) {
    $('status').textContent = err.message === 'abgebrochen' ? 'Abgebrochen.' : 'Fehler: ' + err.message;
    if (err.message !== 'abgebrochen') { $('status').classList.add('error'); console.error(err); }
  }
  clearInterval(tick);
  try { await wakeLock?.release(); } catch { }
  $('go').disabled = false; $('stop').hidden = true;
});
$('stop').addEventListener('click', () => { stopFlag = true; $('stop').disabled = true; setTimeout(() => $('stop').disabled = false, 2000); });
$('share').addEventListener('click', async () => {
  const file = new File([currentBlob], 'handschrift.goodnotes', { type: 'application/octet-stream' });
  try { await navigator.share({ files: [file], title: 'Handschrift' }); } catch (e) { if (e.name !== 'AbortError') $('status').textContent = 'Teilen ging nicht: ' + e.message; }
});

init().catch(e => { $('engine').textContent = 'Fehler beim Start: ' + e.message; $('engine').classList.add('error'); });
