// The model's character table, copied from hw.py (VOCAB; index 0 = padding, 1 = unknown). Append-only.
export const VOCAB = ' !"#%&\'()*+,-./0123456789:;?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz'
  + '$=\\{|}§°²³·¹ÄÖÜßäöüπρ–’“„⁰⁸€−√' + '̈';
export const N_CHARS = VOCAB.length + 2;
const CHAR_ID = new Map([...VOCAB].map((c, i) => [c, i + 2]));
// ä is fed to the generator as a + combining umlaut dots (hw.py UMLAUTS); the readers take ä directly
const UMLAUT = { 'Ä': 'Ä', 'Ö': 'Ö', 'Ü': 'Ü', 'ä': 'ä', 'ö': 'ö', 'ü': 'ü' };

export function encodeText(s) {
  const out = [];
  for (const c of s.replace(/[ÄÖÜäöü]/g, m => UMLAUT[m])) out.push(CHAR_ID.get(c) ?? 1);
  return out;
}

export function readerIds(s) {
  const out = [];
  for (const c of s) out.push(CHAR_ID.get(c) ?? 1);
  return out;
}

export function idToChar(i) {
  return VOCAB[i - 2];
}
