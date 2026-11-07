"""Enter AFTER the bounce stalls instead of a resting limit.
Dump (short): once the bounce has reached >= MINR of the leg, short at the close of
the first 5m bar that closes below the previous bar's low. Stop = bounce high + 0.1*ATR.
Pump (long) mirrored. Targets: T0 = the extreme, T1 = extreme + 0.5*leg beyond,
T2 = trailing (exit when a 5m close is back above the 12-bar EMA after the extreme breaks).
Fees: taker in (0.07% with slippage), maker target 0.02%, taker stop 0.07%.
Also reports the same trade in feature bins, train vs test."""
import numpy as np, pandas as pd
E = pd.read_csv('retr/cont_events_5m.csv', parse_dates=['time'])
df = pd.read_csv('data/okx_btc_5m_trim.csv')
h, l, c = df.h.values, df.l.values, df.c.values; n = len(c)
pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
e12 = pd.Series(c).ewm(span=12).mean().values
import os
TIN, TMK, HZ = (0, 0, 288) if os.environ.get("NOFEE") else (0.0007, 0.0002, 288)

def run(f, MINR, tgt):
    d, ext, leg, j = f['d'], f['ext'], f['leg'], int(f['j'])
    side = d; a = atr[j]
    bh = None
    for k in range(j, min(j + HZ, n - 1)):
        if (side == -1 and l[k] < ext) or (side == 1 and h[k] > ext): return None  # broke before entry
        bh = (max(bh, h[k]) if bh is not None else h[max(j - 1, 0):k + 1].max()) if side == -1 else \
             (min(bh, l[k]) if bh is not None else l[max(j - 1, 0):k + 1].min())
        retr = ((bh - ext) if side == -1 else (ext - bh)) / leg
        if retr > 1.0: return None                                   # bounce went past the start: no setup
        if retr >= MINR and ((side == -1 and c[k] < l[k - 1]) or (side == 1 and c[k] > h[k - 1])):
            entry = c[k]; stop = bh - side * 0.1 * a
            risk = abs(stop - entry) / entry
            T = ext if tgt == 'T0' else ext + side * 0.5 * leg
            broke = False
            for m in range(k + 1, min(k + 3 * HZ, n)):
                if (side == -1 and h[m] >= stop) or (side == 1 and l[m] <= stop):
                    return (-risk - TIN - TIN) / risk
                if tgt in ('T0', 'T1') and ((side == -1 and l[m] <= T) or (side == 1 and h[m] >= T)):
                    return (abs(entry - T) / entry - TIN - TMK) / risk
                if tgt == 'T2':
                    if (side == -1 and l[m] < ext) or (side == 1 and h[m] > ext): broke = True
                    if broke and ((side == -1 and c[m] > e12[m]) or (side == 1 and c[m] < e12[m])):
                        return (side * (c[m] - entry) / entry - 2 * TIN) / risk
            m = min(k + 3 * HZ, n - 1)
            return (side * (c[m] - entry) / entry - 2 * TIN) / risk
    return None

split = E.time.iloc[int(len(E) * 0.6)]
E['half'] = np.where(E.time < split, 'train', 'test')
res = {}
for MINR in [0.236, 0.382, 0.5, 0.618]:
    for tgt in ['T0', 'T1', 'T2']:
        res[(MINR, tgt)] = [run(f, MINR, tgt) for _, f in E.iterrows()]
        E[f'Q_{MINR}_{tgt}'] = res[(MINR, tgt)]
E.to_csv('retr/rejection_events_5m.csv', index=False)
for d, nm in [(-1, 'SHORT after dump'), (1, 'LONG after pump')]:
    S = E[E.d == d]
    print(f'\n{nm}: min bounce, target -> mean R train | test | n test | win test | test 1st half | 2nd half')
    mid = S[S.half == 'test'].time.median()
    for MINR in [0.236, 0.382, 0.5, 0.618]:
        for tgt in ['T0', 'T1', 'T2']:
            col = f'Q_{MINR}_{tgt}'
            a_ = S[S.half == 'train'][col].dropna(); T = S[S.half == 'test']; b_ = T[col].dropna()
            b1 = T[T.time < mid][col].dropna(); b2 = T[T.time >= mid][col].dropna()
            print(f'   {MINR:<6}{tgt}  {a_.mean():+.3f} | {b_.mean():+.3f} | {len(b_):>5} | {(b_ > 0).mean():>4.0%} | {b1.mean():+.3f} | {b2.mean():+.3f}')
