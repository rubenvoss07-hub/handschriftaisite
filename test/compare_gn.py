"""Compares the JavaScript writer's fixtures/out.goodnotes with Python's expected.goodnotes: same entries in
the same order with the same bytes once fresh UUIDs are renamed by first appearance, and every page passes
the vendored strict check. Run after test_layout.mjs:  python web/test/compare_gn.py"""
import re, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor/inkhandwriting/scripts'))
import handwrite as H

FX = ROOT / 'web/test/fixtures'
UUID = re.compile(rb'[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}')


def canon(z):
    seen = {}

    def rep(m):
        u = m.group(0)
        if u not in seen:
            seen[u] = b'<%04d>' % len(seen)
        return seen[u]
    return [(nm, UUID.sub(rep, z.read(nm))) for nm in z.namelist()]


ok = True
with zipfile.ZipFile(FX / 'expected.goodnotes') as a, zipfile.ZipFile(FX / 'out.goodnotes') as b:
    ea, eb = canon(a), canon(b)
    ok &= [n for n, _ in ea] == [n for n, _ in eb]
    print('entries', 'same' if ok else f'differ: {[n for n, _ in ea]} vs {[n for n, _ in eb]}')
    for (na, da), (nb, db) in zip(ea, eb):
        same = da == db
        ok &= same
        print(f'  {na:60s} {len(da):6d} B  {"same" if same else "DIFFERENT"}')
    for nm in b.namelist():
        if nm.startswith('notes/'):
            try:
                H.strict_check_page(b.read(nm))
            except Exception as e:
                ok = False
                print(f'  strict check failed on {nm}: {e}')
print('ALL OK' if ok else 'FAILURES PRESENT')
sys.exit(0 if ok else 1)
