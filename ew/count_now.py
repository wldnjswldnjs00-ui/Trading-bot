"""Swing map from the BTC all-time high to now, at three degrees (percentage zigzag on 1h bars).
Big >= 15%, medium >= 6%, small >= 2.5%. The last pivot is provisional (not yet confirmed)."""
import pandas as pd, numpy as np, sys
df = pd.read_csv(sys.argv[1], parse_dates=['time']).set_index('time')
def zz(x, pct):
    h, l, t = x.h.values, x.l.values, x.index
    piv = [(t[0], h[0], 'H')]; d = -1; ext_i, ext = 0, h[0]
    for i in range(1, len(h)):
        if d == -1:
            if l[i] < ext: ext_i, ext = i, l[i]
            elif h[i] >= ext * (1 + pct): piv.append((t[ext_i], ext, 'L')); d = 1; ext_i, ext = i, h[i]
        else:
            if h[i] > ext: ext_i, ext = i, h[i]
            elif l[i] <= ext * (1 - pct): piv.append((t[ext_i], ext, 'H')); d = -1; ext_i, ext = i, l[i]
    piv.append((t[ext_i], ext, 'L*' if d == -1 else 'H*'))     # provisional
    return piv
ath = df.h.idxmax(); x = df[df.index >= ath]
for name, pct, src in [('BIG (>=15%)', 0.15, x), ('MEDIUM (>=6%)', 0.06, x), ('SMALL (>=2.5%) since the 2026-07 low', 0.025, x[x.index >= x.l.idxmin()])]:
    P = zz(src, pct)
    print(f'\n== {name}')
    for i, (t, p, k) in enumerate(P):
        chg = f'{(p / P[i - 1][1] - 1):+6.1%}' if i else ''
        print(f'  {t:%Y-%m-%d %H:%M}  {k:<2} {p:>10,.0f}  {chg}')
print(f'\nlast price {df.c.iloc[-1]:,.0f} at {df.index[-1]:%Y-%m-%d %H:%M} UTC')
