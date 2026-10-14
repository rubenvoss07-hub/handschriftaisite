"""Writes the fixtures the JavaScript tests compare against: what PyTorch computes for fixed inputs.
Needs torch and the tracked checkpoints. Run from the repo root:  python web/test/dump.py [out_dir]
(default web/test/fixtures/, which stays out of git; the plain weight blob goes there too)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/inkhandwriting/scripts'))
import hw
import gnlib as G
import handwrite as H

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'web/test/fixtures'
OUT.mkdir(parents=True, exist_ok=True)
A = 'Über die Brücke, 52 Mäuse!'
B = 'Das ist ein Test.'
CKPT = 'runs/rl-raft3/avg.pt'


def lst(a):
    return np.asarray(a, np.float32).astype(np.float64).tolist()


def replay(model, text, inputs, prime_text=None, prime_rows=None):
    """Feeds normalized rows through the sampling step exactly as sample_n does (after optionally
    teacher-forcing a priming line) and records what the head and the window produce at every step."""
    s = model.style_vec(torch.tensor([W]), 1, 'cpu')
    enc = hw.encode_text((prime_text + ' ' + text) if prime_text else text)
    U = len(enc)
    onehot = F.one_hot(torch.tensor([enc]), hw.N_CHARS).float()
    onehot[..., 0] = 0
    mask = torch.ones(1, U)
    h = c = torch.zeros(1, model.hidden)
    w, kappa, s2, s3 = torch.zeros(1, hw.N_CHARS), torch.zeros(1, model.K), None, None

    def step(x, h, c, w, kappa, s2, s3):
        h, c = model.lstm1(torch.cat([x, s, w], 1), (h, c))
        w, kappa, phi = model.attend(h, kappa, onehot, mask, extra=1)
        h2, s2 = model.lstm2(torch.cat([x, s, h, w], 1).unsqueeze(1), s2)
        h3, s3 = model.lstm3(torch.cat([x, s, h2[:, 0], w], 1).unsqueeze(1), s3)
        y = model.head(torch.cat([h, h2[:, 0], h3[:, 0]], 1))
        return y, phi, h, c, w, kappa, s2, s3

    with torch.no_grad():
        if prime_rows is not None:
            for row in prime_rows:
                _, _, h, c, w, kappa, s2, s3 = step(torch.tensor(row)[None], h, c, w, kappa, s2, s3)
        ys, phis, kappas, ws = [], [], [], []
        for row in inputs:
            y, phi, h, c, w, kappa, s2, s3 = step(torch.tensor(row)[None], h, c, w, kappa, s2, s3)
            ys.append(y[0]); phis.append(phi[0]); kappas.append(kappa[0]); ws.append(w[0])
        eos, log_pi, mx, my, lsx, lsy, rho = model.split(torch.stack(ys)[:3], bias=1.0)
    return {'ids': enc, 'y': lst(torch.stack(ys)), 'phi': lst(torch.stack(phis)), 'kappa': lst(torch.stack(kappas)),
            'w': lst(torch.stack(ws)),
            'split_bias1': {'eos': lst(eos), 'log_pi': lst(log_pi), 'mx': lst(mx), 'my': lst(my), 'lsx': lst(lsx),
                            'lsy': lst(lsy), 'rho': lst(rho)}}


torch.manual_seed(0)
sd = torch.load(CKPT, map_location='cpu', weights_only=False)
model = hw.model_from_ckpt(sd).eval()
stats = sd['stats']
W = sd['writers']['ruben']
scale = np.array([stats[0], stats[1], 1.0], np.float32)
fx = {'text': A, 'prime_text': B, 'stats': list(stats), 'writer': W, 'vocab': hw.VOCAB, 'n_chars': hw.N_CHARS,
      'encode': {t: hw.encode_text(t) for t in (A, B, 'äöüÄÖÜß °C €', 'ab')},
      'baselines': H.paper_baselines(*G.read_goodnotes(H.SAMPLE)), 'box': H.BOX, 'page_w': H.PAGE_W, 'page_h': H.PAGE_H}

# 1. two sampled versions of A: the raw rows (replay inputs) and the inks (what later stages get)
raw = model.sample_n(A, 2, bias=1.0, seed=3, stats=stats, writer=W, raw=True)
inks = model.sample_n(A, 2, bias=1.0, seed=3, stats=stats, writer=W)
assert all(np.allclose(r[:, :2], i[:, :2]) for r, i in zip(raw, inks))
fx['raw'] = [lst(r) for r in raw]
fx['inks'] = [lst(i) for i in inks]
fx['replay'] = replay(model, A, (raw[0][:-1] / scale).astype(np.float32))
# 2. the same after priming with a written version of B (Graves priming, as best_of_chunks uses it)
prime_ink = model.sample_n(B, 1, bias=1.0, seed=5, stats=stats, writer=W)[0]
raw_p = model.sample_n(A, 1, bias=1.0, seed=7, stats=stats, writer=W, prime=(B, prime_ink), raw=True)[0]
pink = (prime_ink / scale).astype(np.float32)
first = pink[-1].copy()
first[2] = 1
inputs = np.concatenate([first[None], (raw_p[1:-1] / scale).astype(np.float32)])
fx['prime_ink'] = lst(prime_ink)
fx['raw_primed'] = lst(raw_p)
fx['replay_primed'] = replay(model, A, inputs, prime_text=B, prime_rows=pink[:-1])

# 3. the readers on both inks
fx['reader_feats'] = [lst(hw.reader_feats(i)) for i in inks]
fx['readers'] = []
for path in hw._reader_paths(hw.READERS):
    rd = hw.load_reader(path, 'cpu')
    reads, nll, logp, frames = [], [], [], []
    for ink in inks:                 # one ink per batch: padding to a longer ink changes the last frames a little
        rd_reads, rd_nll = hw.reader_run(rd, [ink], [A])
        x, lens = hw._reader_batch([ink], 'cpu')
        with torch.no_grad():
            lp, fl = rd(x, lens)
        reads += rd_reads; nll += lst(rd_nll); logp.append(lst(lp[0, :int(fl[0])])); frames.append(float(fl[0]))
    fx['readers'].append({'path': path, 'cfg': rd.lstm.hidden_size, 'reads': reads, 'nll': nll, 'logp': logp, 'frames': frames})

# 4. layout of ink 0
ink = inks[0]
fx['word_spans'] = [list(map(float, s)) for s in hw.word_spans(ink, A)]
cut = hw.cut_after_words(ink, A, 2)
fx['cut_after_2'] = None if cut is None else len(cut)
wide = hw.widen(ink, A)
fx['widened'] = lst(wide)
strokes = [hw.smooth_stroke(s, 2) for s in hw.to_strokes(wide)]
fx['smoothed'] = [lst(s) for s in strokes]
fx['letter_baseline'] = float(hw.letter_baseline(strokes))
left, right = H.BOX, H.PAGE_W - H.BOX
placed, shrink = hw.place_line(wide, fx['baselines'][1], left, right, hw.GN_SIZE, 2)
fx['placed'] = {'strokes': [lst(s) for s in placed], 'shrink': float(shrink)}

# 5. a two-page document from placed strokes, written with predictable UUIDs so the files can be compared
counter = [0]


def fake_uuid():
    counter[0] += 1
    return '%08X-0000-4000-8000-%012X' % (counter[0], counter[0])


G.gn_uuid = fake_uuid
p1 = placed + hw.place_line(hw.widen(inks[1], A), fx['baselines'][2], left, right, hw.GN_SIZE, 2)[0]
p2 = hw.place_line(wide, fx['baselines'][0], left, right, hw.GN_SIZE, 2)[0]
pages = [p1, p2]
fx['gn_pages'] = [[lst(s) for s in pg] for pg in pages]
hw.write_goodnotes(pages, OUT / 'expected.goodnotes')
fx['gn_uuids_used'] = counter[0]

(OUT / 'fixtures.json').write_text(json.dumps(fx), encoding='utf-8')
subprocess.run([sys.executable, str(ROOT / 'hw.py'), 'pack', '--plain', '--out', str(OUT / 'plain.bin')], check=True)
print(f'wrote {OUT}: fixtures.json, expected.goodnotes ({counter[0]} uuids), plain.bin; '
      f'{len(raw[0])} + {len(raw[1])} rows, primed {len(raw_p)} rows')
