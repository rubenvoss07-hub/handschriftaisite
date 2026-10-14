// Writes a .goodnotes file: the vendored inkhandwriting scripts (gnlib.py, gnwrite.py, handwrite.py) ported
// to JavaScript, same bytes. A page is a stream of [varint length][protobuf]; two messages per stroke.
const te = new TextEncoder();

// --- bytes
export function concatBytes(parts) {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
const bytesToStr = b => { let s = ''; for (let i = 0; i < b.length; i += 8192) s += String.fromCharCode(...b.subarray(i, i + 8192)); return s; };
const strToBytes = s => { const b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b; };
const u32le = v => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); return b; };

// --- varint / protobuf
export function readVarint(b, i, end = b.length) {
  let shift = 0, result = 0;
  const start = i;
  for (;;) {
    if (i >= end) throw new Error(`truncated varint at ${start}`);
    const byte = b[i++];
    result += (byte & 0x7f) * 2 ** shift;
    if (!(byte & 0x80)) break;
    shift += 7;
    if (shift > 63) throw new Error(`varint too long at ${start}`);
  }
  return [result, i];
}
export function writeVarint(v) {
  const out = [];
  for (;;) {
    const b = v % 128;
    v = Math.floor(v / 128);
    out.push(b | (v ? 0x80 : 0));
    if (!v) break;
  }
  return Uint8Array.from(out);
}
/** [field, wiretype, keyStart, valStart, valEnd] for every top-level field. */
export function* pbFields(buf) {
  let i = 0;
  while (i < buf.length) {
    const keyStart = i;
    let tag;
    [tag, i] = readVarint(buf, i);
    const fnum = Math.floor(tag / 8), wt = tag & 7;
    if (wt === 0) [, i] = readVarint(buf, i);
    else if (wt === 1) i += 8;
    else if (wt === 2) { let ln; [ln, i] = readVarint(buf, i); yield [fnum, wt, keyStart, i, i + ln]; i += ln; continue; }
    else if (wt === 5) i += 4;
    else throw new Error(`bad wiretype ${wt} at ${keyStart}`);
    yield [fnum, wt, keyStart, keyStart, i];
  }
}
export const pbKey = (fnum, wt) => writeVarint(fnum * 8 + wt);
const lenField = (fnum, val) => concatBytes([pbKey(fnum, 2), writeVarint(val.length), val]);
const strField = (fnum, s) => lenField(fnum, te.encode(s));
const f32Field = (fnum, v) => { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); return concatBytes([pbKey(fnum, 5), b]); };

function replaceField(buf, fnum, newField, test = () => true) {
  for (const [f, w, ks, vs, ve] of pbFields(buf)) if (f === fnum && w === 2 && test(vs, ve)) return concatBytes([buf.subarray(0, ks), newField, buf.subarray(ve)]);
  return null;
}

/** SwiftProtobuf-style strictness for one message (strictpb.parse at the top level). */
export function strictParse(b) {
  let i = 0;
  const end = b.length;
  while (i < end) {
    const koff = i;
    let tag;
    [tag, i] = readVarint(b, i, end);
    const fnum = Math.floor(tag / 8), wt = tag & 7;
    if (fnum === 0) throw new Error(`field number 0 at ${koff}`);
    if (wt === 3 || wt === 4 || wt === 6 || wt === 7) throw new Error(`bad wiretype ${wt} at ${koff}`);
    if (wt === 0) [, i] = readVarint(b, i, end);
    else if (wt === 1) { if (i + 8 > end) throw new Error(`truncated 64bit at ${koff}`); i += 8; }
    else if (wt === 5) { if (i + 4 > end) throw new Error(`truncated 32bit at ${koff}`); i += 4; }
    else { let ln; [ln, i] = readVarint(b, i, end); if (i + ln > end) throw new Error(`field ${fnum} overruns at ${koff}`); i += ln; }
  }
}

// --- streams of messages
export function splitStream(buf) {
  const msgs = [];
  let i = 0;
  while (i < buf.length) { let ln; [ln, i] = readVarint(buf, i); msgs.push(buf.subarray(i, i + ln)); i += ln; }
  if (i !== buf.length) throw new Error('stream overran');
  return msgs;
}
export const buildStream = msgs => concatBytes(msgs.flatMap(m => [writeVarint(m.length), m]));

// --- the stroke geometry: tpl record inside an all-literal LZ4 "bv41" frame
const SCHEMA = 'vuA(v)A(S(uu))A(S(uuuu))vA(f)';
export function buildTpl(points, width) {
  if (points.length % 2) throw new Error('point count must be even');
  const ns = points.length / 2;
  const size = 4 + 4 + SCHEMA.length + 1 + 2 + 4 + 4 + 2 * (ns + 1) + 4 + 8 + 4 + 16 * ns + 2 + 4;
  const b = new Uint8Array(size), dv = new DataView(b.buffer);
  let p = 0;
  b.set(te.encode('tpl\0'), p); p += 4;
  dv.setUint32(p, size, true); p += 4;
  b.set(te.encode(SCHEMA), p); p += SCHEMA.length; b[p++] = 0;
  dv.setUint16(p, 2, true); p += 2;
  dv.setFloat32(p, width, true); p += 4;
  dv.setUint32(p, ns + 1, true); p += 4;
  dv.setUint16(p, 0, true); p += 2;
  for (let k = 0; k < ns; k++) { dv.setUint16(p, 1, true); p += 2; }
  dv.setUint32(p, 1, true); p += 4;
  dv.setFloat32(p, points[0][0], true); p += 4; dv.setFloat32(p, points[0][1], true); p += 4;
  dv.setUint32(p, ns, true); p += 4;
  for (let k = 0; k < ns; k++) for (const v of [points[2 * k][0], points[2 * k][1], points[2 * k + 1][0], points[2 * k + 1][1]]) { dv.setFloat32(p, v, true); p += 4; }
  dv.setUint16(p, 1, true); p += 2;
  dv.setUint32(p, 0, true); p += 4;
  if (p !== size) throw new Error('tpl size');
  return b;
}
function lz4Literal(data) {
  const n = data.length, head = [(n >= 15 ? 15 : n) << 4];
  if (n >= 15) { let rem = n - 15; while (rem >= 255) { head.push(255); rem -= 255; } head.push(rem); }
  return concatBytes([Uint8Array.from(head), data]);
}
export function bv41Frame(dec) {
  const payload = lz4Literal(dec);
  return concatBytes([te.encode('bv41'), u32le(dec.length), u32le(payload.length), payload, te.encode('bv4$')]);
}
const isBv41 = (buf, vs) => buf[vs] === 0x62 && buf[vs + 1] === 0x76 && buf[vs + 2] === 0x34 && buf[vs + 3] === 0x31;

export function field7Unwrap(geom) {
  for (const [f, w, , vs, ve] of pbFields(geom)) if (f === 7 && w === 2) return geom.subarray(vs, ve);
  throw new Error('geom msg has no field 7');
}
export const field7Wrap = body => lenField(7, body);
export function bodyReplaceBv41(body, nb) {
  const r = replaceField(body, 2, lenField(2, nb), vs => isBv41(body, vs));
  if (!r) throw new Error('no bv41 field 2');
  return r;
}
export function bodySetUuid(body, u) { const r = replaceField(body, 1, strField(1, u)); if (!r) throw new Error('no field 1'); return r; }
export const headerSetUuid = (h, u) => replaceField(h, 1, strField(1, u)) ?? h;
export function bodySetColor(body, r, g, b, a = 1.0) {
  const sub = concatBytes([f32Field(1, r), f32Field(2, g), f32Field(3, b), f32Field(4, a)]), nf = lenField(4, sub);
  return replaceField(body, 4, nf) ?? concatBytes([body, nf]);
}

// --- UUIDs and the template's per-page events (handwrite.py)
const HEX = '0123456789ABCDEF';
export const gnUuid = () => crypto.randomUUID().toUpperCase();
const UUID_RE = /[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}/g;
const pageSibling = (u, delta) => u.slice(0, -1) + HEX[(HEX.indexOf(u[u.length - 1].toUpperCase()) + delta + 16) % 16];
function newPagePair(uuid) {
  for (;;) { const u = uuid(); if (u[u.length - 1].toUpperCase() !== 'F') return [u, pageSibling(u, 1)]; }
}
const uuidsIn = s => [...new Set(s.match(UUID_RE) || [])];
function docUuid(evs) {
  const c = new Map();
  for (const m of evs) for (const u of uuidsIn(m)) c.set(u, (c.get(u) || 0) + 1);
  let best = null;
  for (const [u, n] of c) if (best === null || n > c.get(best)) best = u;
  return best;
}
function paperUuid(evs) {
  for (const m of evs) {
    if (!m.includes('_standard') && !m.includes('tandard')) continue;
    for (const [f, w, , vs, ve] of pbFields(strToBytes(m))) if (f === 1 && w === 2 && ve - vs === 36) return m.slice(vs, ve);
  }
  return null;
}
function cloneEvents(events, remap, keep, uuid) {
  return events.map(m => {
    let s = m;
    for (const u of uuidsIn(m)) {
      if (keep.has(u)) continue;
      s = s.replaceAll(u, remap.get(u) ?? uuid());
    }
    return s;
  });
}
function orderedNotePaths(entries) {
  const paths = [];
  for (const rec of splitStream(entries.get('index.notes.pb'))) for (const [f, w, , vs, ve] of pbFields(rec)) if (f === 2 && w === 2) paths.push(bytesToStr(rec.subarray(vs, ve)));
  return paths;
}
const notesIndexRecord = u => concatBytes([strField(1, u), strField(2, 'notes/' + u)]);

// --- zip (the template in, the document out)
const CRC = new Int32Array(256);
for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; CRC[n] = c; }
function crc32(b) { let c = -1; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ -1) >>> 0; }
async function pipe(data, stream) {
  const out = new Blob([data]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(out).arrayBuffer());
}
export async function readZip(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let e = bytes.length - 22;
  while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new Error('not a zip');
  const count = dv.getUint16(e + 10, true);
  let p = dv.getUint32(e + 16, true);
  const entries = new Map(), names = [];
  for (let k = 0; k < count; k++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('bad central header');
    const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true), nlen = dv.getUint16(p + 28, true);
    const elen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true), loc = dv.getUint32(p + 42, true);
    const name = bytesToStr(bytes.subarray(p + 46, p + 46 + nlen));
    const start = loc + 30 + dv.getUint16(loc + 26, true) + dv.getUint16(loc + 28, true);
    const raw = bytes.subarray(start, start + csize);
    entries.set(name, method === 8 ? await pipe(raw, new DecompressionStream('deflate-raw')) : Uint8Array.from(raw));
    names.push(name);
    p += 46 + nlen + elen + clen;
  }
  return { entries, names };
}
export async function writeZip(entries, names) {
  const parts = [], central = [];
  let offset = 0;
  const rec = (sig, fields) => { const b = new Uint8Array(fields.reduce((n, [w]) => n + w, 4)), dv = new DataView(b.buffer); dv.setUint32(0, sig, true); let p = 4; for (const [w, v] of fields) { if (w === 2) dv.setUint16(p, v, true); else dv.setUint32(p, v, true); p += w; } return b; };
  for (const name of names) {
    const data = entries.get(name), nb = te.encode(name);
    const method = name.endsWith('/') || data.length === 0 ? 0 : 8;
    const comp = method ? await pipe(data, new CompressionStream('deflate-raw')) : data;
    const crc = crc32(data), date = 0x21, time = 0;   // 1980-01-01, like Python's ZipInfo default
    const common = [[2, 20], [2, 0], [2, method], [2, time], [2, date], [4, crc], [4, comp.length], [4, data.length], [2, nb.length]];
    parts.push(rec(0x04034b50, [...common, [2, 0]]), nb, comp);
    central.push(rec(0x02014b50, [[2, 20], ...common, [2, 0], [2, 0], [2, 0], [2, 0], [4, 0], [4, offset]]), nb);
    offset += 30 + nb.length + comp.length;
  }
  const cd = concatBytes(central);
  parts.push(cd, rec(0x06054b50, [[2, 0], [2, 0], [2, names.length], [2, names.length], [4, cd.length], [4, offset], [2, 0]]));
  return concatBytes(parts);
}

// --- the document
function makeStrokePair(header, body, points, width, uid, color) {
  const xy = points.length % 2 ? [...points, points[points.length - 1]] : points;
  const bv41 = bv41Frame(buildTpl(xy, width));
  let b = bodyReplaceBv41(body, bv41);
  b = bodySetUuid(b, uid);
  b = bodySetColor(b, ...color);
  return [headerSetUuid(header, uid), field7Wrap(b)];
}

/** pages: lists of strokes ([[x, y], ...] in page units) -> the .goodnotes bytes. Page 0 overwrites the
 *  template's single lined page; more pages clone its events, keeping the doc and paper UUIDs. */
export async function writeGoodnotes(pages, { template, proto, uuid = gnUuid, color = [0.902, 0.451, 0.0], width = 2.25 }) {
  const { entries, names } = await readZip(template);
  const [tplHeader, tplGeom] = splitStream(proto), tplBody = field7Unwrap(tplGeom);
  const mk = strokes => buildStream(strokes.flatMap(s => makeStrokePair(tplHeader, tplBody, s, width, uuid(), color)));
  const notePaths = orderedNotePaths(entries), pg0Path = notePaths[0], pg0Content = pg0Path.split('/').slice(1).join('/');
  const pg0Entity = pageSibling(pg0Content, -1);
  const evMsgs = splitStream(entries.get('index.events.pb')).map(bytesToStr);
  const doc = docUuid(evMsgs), paper = paperUuid(evMsgs);
  const pageEvents = evMsgs.filter(m => m.includes(pg0Content) || m.includes(pg0Entity));
  entries.set(pg0Path, mk(pages[0]));
  const used = [pg0Path];
  if (pages.length > 1) {
    if (!pageEvents.length) throw new Error('template has no per-page events to clone');
    const idx = splitStream(entries.get('index.notes.pb'));
    const keep = new Set([doc, paper].filter(Boolean));
    for (const strokes of pages.slice(1)) {
      const [ent, con] = newPagePair(uuid);
      evMsgs.push(...cloneEvents(pageEvents, new Map([[pg0Entity, ent], [pg0Content, con]]), keep, uuid));
      const path = 'notes/' + con;
      entries.set(path, mk(strokes));
      names.push(path);
      idx.push(notesIndexRecord(con));
      used.push(path);
    }
    entries.set('index.events.pb', buildStream(evMsgs.map(strToBytes)));
    entries.set('index.notes.pb', buildStream(idx));
  }
  for (const path of used) for (const m of splitStream(entries.get(path))) strictParse(m);   // GoodNotes rejects the whole import on one malformed message
  for (const m of evMsgs) [...pbFields(strToBytes(m))];
  return writeZip(entries, names);
}
