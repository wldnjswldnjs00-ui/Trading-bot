"""Does price come back to the level where volume FIRST exploded?

Impulse (causal): price moves >= 4*ATR within 6 bars.
Surge bar: the FIRST bar, from 12 bars before the impulse start up to the trigger,
whose volume >= M x the average volume of the 24 bars before that window (= quiet volume).
Surge level (the yellow line), 3 versions:
  close  = close of the surge bar
  edge   = high of the surge bar for a dump / low for a pump (the 'top of the candle')
  vwap   = (h+l+c)/3 of the surge bar
Bottom/top of the move is confirmed causally (1.5 ATR bounce or 48 bars).
Measured from that confirmation: did price touch the level within 24/72/168 bars,
and did it touch it BEFORE breaking the low (high) of the move?"""
import numpy as np, pandas as pd, sys
tf = sys.argv[1] if len(sys.argv) > 1 else '1h'
df = pd.read_csv('data/btcusdt_1h.csv', parse_dates=['time']).set_index('time')
if tf != '1h':
    df = df.resample(tf).agg(dict(o='first', h='max', l='min', c='last', v='sum')).dropna()
h, l, c, v = (df[k].values for k in 'hlcv')
n = len(c); pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
K, CONF, MAXLEG, HS = 4.0, 1.5, 48, [24, 72, 168]

def run(M):
    rows, t, last = [], 800, 0
    while t < n - 200:
        ev = None
        for d in (-1, 1):
            win = c[t - 6:t]
            bi = t - 6 + (np.argmax(win) if d == -1 else np.argmin(win))
            if bi <= last or np.isnan(atr[bi]) or (c[t] - c[bi]) * d < K * atr[bi]:
                continue
            w0 = max(bi - 12, last + 1)
            quiet = v[w0 - 24:w0].mean()
            idx = [k for k in range(w0, t + 1) if v[k] >= M * quiet]
            if not idx: continue
            s = idx[0]
            # the surge bar must be on the move's side (not after the move is done)
            ev = dict(d=d, t=t, s=s, time=df.index[s], close=c[s],
                      edge=h[s] if d == -1 else l[s], vwap=(h[s] + l[s] + c[s]) / 3,
                      lead=t - s)
            break
        if not ev: t += 1; continue
        d, a = ev['d'], atr[ev['s']]
        ext, ei, j = (l[t], t, t + 1) if d == -1 else (h[t], t, t + 1)
        while j < n and j - t <= MAXLEG:
            if d == -1:
                if l[j] < ext: ext, ei = l[j], j
                elif h[j] - ext >= CONF * a: break
            else:
                if h[j] > ext: ext, ei = h[j], j
                elif ext - l[j] >= CONF * a: break
            j += 1
        ev['ext'] = ext
        for lv in ('close', 'edge', 'vwap'):
            L = ev[lv]
            ev[f'dist_{lv}'] = abs(L - ext) / abs(c[ev['s']] - ext + 1e-9)
            for H in HS:
                hh, ll = h[ei:j + H], l[ei:j + H]
                touch = (hh >= L) if d == -1 else (ll <= L)
                brk = (ll < ext) if d == -1 else (hh > ext)
                ti = np.argmax(touch) if touch.any() else None
                bi_ = np.argmax(brk) if brk.any() else None
                ev[f'{lv}_touch_{H}'] = ti is not None
                ev[f'{lv}_first_{H}'] = ti is not None and (bi_ is None or ti <= bi_)
                if H == 168: ev[f'{lv}_bars'] = ti if ti is not None else np.nan
        rows.append(ev); last = j; t = j + 1
    return pd.DataFrame(rows)

for M in [2, 3]:
    E = run(M)
    E.to_csv(f'retr/surge_events_{tf}_M{M}.csv', index=False)
    for d, nm in [(-1, 'DUMP -> does it come back UP to the surge level?'),
                  (1, 'PUMP -> does it come back DOWN to the surge level?')]:
        S = E[E.d == d]
        print(f'\n[{tf}, volume >= {M}x quiet] {nm}   n={len(S)}')
        print(f"  {'level':>6} | {'24h':>5} {'72h':>5} {'7d':>5} | {'before new low/high (7d)':>24} | {'median hrs':>10}")
        rng = np.random.default_rng(0)
        for lv in ('close', 'edge', 'vwap'):
            x = S[f'{lv}_first_168'].values.astype(float)
            bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
            hrs = S[f'{lv}_bars'].median() * (1 if tf == '1h' else int(tf[:-1]))
            print(f"  {lv:>6} | " + ' '.join(f"{S[f'{lv}_touch_{H}'].mean():>5.0%}" for H in HS)
                  + f" | {x.mean():>9.0%} [{np.percentile(bs, 2.5):.0%}-{np.percentile(bs, 97.5):.0%}]     | {hrs:>10.0f}")
        ts = S.time.astype(str)
        for a_, b_ in [('2017', '2021-12-31'), ('2022', '2026-12-31')]:
            P = S[(ts >= a_) & (ts <= b_)]
            print(f"   {a_}-{b_[:4]} n={len(P):>3}: close {P['close_first_168'].mean():.0%} / edge {P['edge_first_168'].mean():.0%} before new extreme; "
                  f"touch within 7d close {P['close_touch_168'].mean():.0%}")
