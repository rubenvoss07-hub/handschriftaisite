// Drives index.html like a user: wrong password, right password, write two lines at "Schnell", check
// the result, hand the .goodnotes file back to the harness. Run with:
//   node web/test/run.mjs 'index.html?model=test/fixtures/model-test.enc' --drive web/test/drive_page.js --shot
window.__log = window.__log || [];
let fails = 0;
const log = s => window.__log.push(s);
const check = (name, ok, extra = '') => { if (!ok) fails++; log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${extra}`); };
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const until = async (f, ms = 600000) => { const t0 = Date.now(); while (!f()) { if (Date.now() - t0 > ms) throw new Error('timeout waiting'); await sleep(250); } };
await until(() => $('engine').textContent !== '…');
log('engine: ' + $('engine').textContent);
$('pw').value = 'falsch'; $('unlock').click();
await until(() => $('loginStatus').textContent.includes('Passwort') || $('loginStatus').textContent.startsWith('Fehler'), 120000);
check('wrong password is refused', $('loginStatus').textContent === 'Falsches Passwort.', $('loginStatus').textContent);
$('pw').value = 'test'; $('unlock').click();
await until(() => !$('write').hidden, 180000);
check('right password unlocks', !$('write').hidden);
$('text').value = 'Hallo Welt.\nDas ist ein Test.';
$('quality').value = 'fast';
$('seed').value = '3';
const t0 = Date.now();
$('go').click();
await until(() => !$('result').hidden || $('status').classList.contains('error'), 1500000);
check('a page was written', !$('result').hidden, $('status').textContent);
log(`status: ${$('status').textContent} (${Math.round((Date.now() - t0) / 1000)} s)`);
log('lines: ' + [...$('lines').children].map(li => li.textContent).join(' | '));
check('two lines', $('lines').children.length === 2);
check('preview drawn', $('preview').querySelectorAll('canvas').length === 1);
check('download link is a blob', $('download').href.startsWith('blob:'));
const blob = await (await fetch($('download').href)).blob();
check('file has content', blob.size > 2000, `${blob.size} bytes`);
const b64 = btoa(String.fromCharCode(...new Uint8Array(await blob.arrayBuffer())));
window.__file = { name: 'page.goodnotes', base64: b64 };
log(fails ? `${fails} FAILED` : 'ALL OK');
window.__done = { fails };
