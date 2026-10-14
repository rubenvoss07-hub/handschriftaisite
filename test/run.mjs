// Serves web/ on 127.0.0.1 and opens a page in headless Chromium (WebGPU on, software-rendered) over the
// DevTools protocol; prints what the page logs to window.__log until it sets window.__done.
// Usage: node web/test/run.mjs test/gpu.html [--chrome PATH] [--drive script.js] [--shot] [--keep]
// --drive evaluates a script inside the page once it has loaded (it must set window.__done at the end);
// a window.__file = {name, base64} is saved to web/test/out/.
import { spawn } from 'node:child_process';
import fs from 'node:fs'; import path from 'node:path'; import os from 'node:os'; import http from 'node:http';

const here = path.dirname(new URL(import.meta.url).pathname), root = path.join(here, '..');
const args = process.argv.slice(2), page = args.find(a => !a.startsWith('--')) || 'test/gpu.html';
const chrome = args.includes('--chrome') ? args[args.indexOf('--chrome') + 1]
  : [process.env.CHROME, '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(p => p && fs.existsSync(p));
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.png': 'image/png', '.bin': 'application/octet-stream', '.enc': 'application/octet-stream', '.goodnotes': 'application/octet-stream' };
const srv = http.createServer((q, r) => {
  const f = path.join(root, decodeURIComponent(q.url.split('?')[0]));
  if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end('not found'); }
  r.setHeader('Content-Type', MIME[path.extname(f)] || 'application/octet-stream');
  fs.createReadStream(f).pipe(r);
}).listen(0, '127.0.0.1');
await new Promise(r => srv.on('listening', r));
const url = `http://127.0.0.1:${srv.address().port}/${page}`;
const prof = fs.mkdtempSync(path.join(os.tmpdir(), 'hwweb-')), dl = path.join(here, 'out');
fs.mkdirSync(dl, { recursive: true });
const proc = spawn(chrome, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${prof}`, '--no-sandbox', '--no-first-run', '--mute-audio',
  '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--enable-features=Vulkan,WebGPU', '--window-size=1024,1366', 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let port; for (let i = 0; i < 100 && !port; i++) { try { port = fs.readFileSync(path.join(prof, 'DevToolsActivePort'), 'utf8').split('\n')[0]; } catch { await sleep(100); } }
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.onopen = r);
let id = 0; const pending = new Map();
ws.onmessage = m => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
  if (d.method === 'Runtime.exceptionThrown') console.log('EXC ' + (d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text));
  if (d.method === 'Log.entryAdded' && ['error', 'warning'].includes(d.params.entry.level)) console.log('LOG ' + d.params.entry.text.slice(0, 400));
  if (d.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(d.params.type)) console.log(d.params.type.toUpperCase() + ' ' + d.params.args.map(a => a.value ?? a.description).join(' '));
};
const cdp = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await cdp('Runtime.enable'); await cdp('Page.enable'); await cdp('Log.enable');
await cdp('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dl });
const loaded = new Promise(r => { const h = ws.onmessage; ws.onmessage = m => { h(m); if (JSON.parse(m.data).method === 'Page.loadEventFired') r(); }; });
await cdp('Page.navigate', { url });
if (args.includes('--drive')) {
  await loaded; await sleep(300);
  const script = fs.readFileSync(path.resolve(args[args.indexOf('--drive') + 1]), 'utf8');
  cdp('Runtime.evaluate', { expression: `(async () => { ${script} })().catch(e => { (window.__log = window.__log || []).push('DRIVE ERROR ' + (e.stack || e)); window.__done = { fails: 1 }; })`, awaitPromise: false });
}
let printed = 0, done = false;
const t0 = Date.now();
while (!done && Date.now() - t0 < 30 * 60 * 1000) {
  await sleep(500);
  const r = await cdp('Runtime.evaluate', { expression: 'JSON.stringify({log: window.__log || [], done: !!window.__done})', returnByValue: true });
  const v = r.result?.result?.value;
  if (!v) continue;
  const { log, done: d } = JSON.parse(v);
  for (; printed < log.length; printed++) console.log(log[printed]);
  done = d;
}
const r = await cdp('Runtime.evaluate', { expression: 'JSON.stringify(window.__done || null)', returnByValue: true });
const result = JSON.parse(r.result?.result?.value || 'null');
const f = await cdp('Runtime.evaluate', { expression: 'JSON.stringify(window.__file || null)', returnByValue: true });
const file = JSON.parse(f.result?.result?.value || 'null');
if (file) { fs.writeFileSync(path.join(dl, file.name), Buffer.from(file.base64, 'base64')); console.log(`saved ${path.join(dl, file.name)}`); }
if (args.includes('--shot')) { const s = await cdp('Page.captureScreenshot', { format: 'png' }); fs.writeFileSync(path.join(dl, 'page.png'), Buffer.from(s.result.data, 'base64')); }
if (!args.includes('--keep')) { proc.kill(); srv.close(); } else console.log(`kept open at ${url}`);
if (!done) { console.log('TIMEOUT'); process.exit(2); }
process.exit(result && result.fails ? 1 : 0);
