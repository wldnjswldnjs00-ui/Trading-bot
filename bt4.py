"""v4: v3 setup + confluence score (fib, 매물대, old S/R). Swing or scalp mode.

usage: python bt4.py data/btcusdt_1h.csv swing
       python bt4.py data/btcusdt_5m.csv scalp
Scalp: fixed target in R, time stop, futures-like fees by default.
"""
import sys, bisect, itertools, numpy as np, pandas as pd
from engine import atr, zigzag
from confluence import fib_hit, vp_strength, sr_hit

path, mode = (sys.argv + ['data/btcusdt_1h.csv', 'swing'])[1:3]
df = pd.read_csv(path); df['time'] = pd.to_datetime(df.time, format='ISO8601', utc=True)
A = atr(df); o, h, l, c, v = (df[k].values for k in 'ohlcv')
SPLIT = pd.Timestamp('2023-01-01', tz='UTC'); RISK = 0.01; BUF = 0.001
VPWIN = 720 if mode == 'swing' else 2016          # 30 days of 1h / 7 days of 5m


def setups(K, ratio, zone, maxstop, tol):
    """All candidate entries with their confluence features (no exits yet)."""
    big = zigzag(df, K, A); low = zigzag(df, K / ratio, A); out = []
    lk = [q[3] for q in low]; bk = [p[3] for p in big]
    for j in range(1, len(big)):
        p0, p1 = big[j - 1], big[j]; up = p1[2] == 1; s = 1 if up else -1
        w1 = s * (p1[1] - p0[1])
        if w1 <= 0: continue
        nxt = big[j + 1][3] if j + 1 < len(big) else len(o)
        for q in low[bisect.bisect_right(lk, p1[3]):]:
            if q[3] >= nxt: break
            if q[2] != (-1 if up else 1) or q[0] <= p1[0]: continue
            r = s * (p1[1] - q[1]) / w1
            if s * (q[1] - p0[1]) <= 0: break                 # wave 2 beyond start of 1
            if not (zone[0] <= r < zone[1]): continue
            e = q[3] + 1
            if e >= len(o): break
            entry = o[e]; stop = q[1] * (1 - s * BUF); risk = s * (entry - stop) / entry
            if risk <= 0 or risk > maxstop: continue
            f = fib_hit(q[1], p0[1], p1[1], tol)
            vp = vp_strength(h, l, v, max(0, p1[0] - VPWIN), p1[0], q[1], tol)
            sr = sr_hit(big[max(0, bisect.bisect_left(bk, p0[3]) - 60):bisect.bisect_left(bk, p0[3])], p0[3], q[1], tol)
            out.append(dict(e=e, up=up, entry=entry, stop=stop, risk=risk, base=q[1], w1=w1,
                            fib=f, vp=int(vp >= 1.5), sr=sr))
            break
    return out


def simulate(S, tgt_mode, tgt, fee, tstop):
    res = []; busy = -1
    for x in S:
        e = x['e']
        if e <= busy: continue
        s = 1 if x['up'] else -1; entry, stop = x['entry'], x['stop']
        T = x['base'] + s * tgt * x['w1'] if tgt_mode == 'ext' else entry + s * tgt * x['risk'] * entry
        end = min(len(o), e + tstop) if tstop else len(o)
        px = None
        for t in range(e, end):
            if (l[t] <= stop) if s > 0 else (h[t] >= stop): px = stop; break
            if (h[t] >= T) if s > 0 else (l[t] <= T): px = T; break
        if px is None: t = end - 1; px = c[t]
        R = (s * (px - entry) / entry - fee) / x['risk']
        res.append(dict(time=df.time[e], R=R, score=x['fib'] + x['vp'] + x['sr'], **{k: x[k] for k in ('fib', 'vp', 'sr')}))
        busy = t
    return pd.DataFrame(res)


def summary(T):
    if len(T) < 5: return dict(n=len(T))
    eq = np.cumprod(1 + RISK * T.R.values)
    streak = m = 0
    for r in T.R.values:
        streak = streak + 1 if r <= 0 else 0; m = max(m, streak)
    return dict(n=len(T), win=(T.R > 0).mean(), avgR=T.R.mean(), total=eq[-1] - 1,
                mdd=(eq / np.maximum.accumulate(eq) - 1).min(), streak=m)


if __name__ == '__main__':
    if mode == 'swing':
        grid = itertools.product([4, 6], [3], [(0.382, 0.886)], [0.02], [0.004],
                                 [('ext', 1.618), ('ext', 2.618)], [0.0012], [0])
    else:
        grid = itertools.product([4, 6, 8], [3], [(0.382, 0.886)], [0.004, 0.007, 0.01], [0.002],
                                 [('R', 1.5), ('R', 2.0), ('R', 3.0)], [0.0004, 0.0012], [48, 144])
    rows = []
    cache = {}
    for K, ra, z, ms, tol, (tm, tg), fee, ts in grid:
        key = (K, ra, z, ms, tol)
        if key not in cache: cache[key] = setups(K, ra, z, ms, tol)
        T = simulate(cache[key], tm, tg, fee, ts)
        if T.empty: continue
        for name, sub in [('all', T), ('score>=1', T[T.score >= 1]), ('score>=2', T[T.score >= 2]),
                          ('fib+vp', T[(T.fib == 1) & (T.vp == 1)])]:
            a = summary(sub[sub.time < SPLIT]); b = summary(sub[sub.time >= SPLIT])
            rows.append(dict(K=K, maxstop=ms, tgt=f'{tm}{tg}', fee=fee, tstop=ts, filt=name,
                             **{f'{k}_in': v for k, v in a.items()}, **{f'{k}_out': v for k, v in b.items()}))
    R = pd.DataFrame(rows); R.to_csv(f'grid4_{mode}.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))
