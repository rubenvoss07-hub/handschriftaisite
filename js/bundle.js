// The model bundle: web/model.enc = HWB1 + header JSON + AES-256-GCM ciphertext of the plain blob
// (HWM1 + manifest JSON + float32 data). Written by `hw.py pack`; opened here with the password.
const te = new TextEncoder(), td = new TextDecoder();

function u32(buf, off) {
  return new DataView(buf.buffer, buf.byteOffset + off, 4).getUint32(0, true);
}

function b64(s) {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Decrypts the bundle bytes with the password. Throws on a wrong password (the GCM tag fails). */
export async function unlock(bytes, password, subtle = globalThis.crypto.subtle) {
  if (td.decode(bytes.subarray(0, 4)) !== 'HWB1') throw new Error('not a model bundle');
  const n = u32(bytes, 4);
  const hdr = JSON.parse(td.decode(bytes.subarray(8, 8 + n)));
  const ct = bytes.subarray(8 + n);
  const base = await subtle.importKey('raw', te.encode(password), 'PBKDF2', false, ['deriveKey']);
  const key = await subtle.deriveKey({ name: 'PBKDF2', hash: 'SHA-256', salt: b64(hdr.salt), iterations: hdr.iterations },
    base, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
  let plain;
  try {
    plain = await subtle.decrypt({ name: 'AES-GCM', iv: b64(hdr.nonce) }, key, ct);
  } catch (e) {
    throw new Error('wrong password');
  }
  return parsePlain(new Uint8Array(plain));
}

/** The plain blob -> { meta, tensor(name) -> {shape, data: Float32Array view} } */
export function parsePlain(bytes) {
  if (td.decode(bytes.subarray(0, 4)) !== 'HWM1') throw new Error('not a weight blob');
  const n = u32(bytes, 4);
  const meta = JSON.parse(td.decode(bytes.subarray(8, 8 + n)));
  const start = 8 + n;
  // float32 views need 4-byte alignment: copy once if the data does not start on one
  let data;
  if ((bytes.byteOffset + start) % 4 === 0) data = new Float32Array(bytes.buffer, bytes.byteOffset + start, (bytes.length - start) >> 2);
  else data = new Float32Array(bytes.slice(start).buffer);
  const tensor = (table, name) => {
    const t = table[name];
    if (!t) throw new Error('missing tensor ' + name);
    const size = t.shape.reduce((a, b) => a * b, 1);
    return { shape: t.shape, data: data.subarray(t.offset, t.offset + size) };
  };
  return {
    meta,
    generator: { ...meta.generator, get: name => tensor(meta.generator.tensors, name) },
    readers: meta.readers.map(r => ({ ...r, get: name => tensor(r.tensors, name) })),
  };
}
