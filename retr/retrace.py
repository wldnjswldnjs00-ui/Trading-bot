"""After a volume-surge impulse, how far does price retrace back toward the base?

Event (causal, all known at the trigger bar t):
  base  = highest close (down move) / lowest close (up move) of the 6 bars before t
  move  = base -> close[t] >= K * ATR(100)               (big candle(s))
  quiet = mean vol of the 24 bars before the base <= Q * median vol of 720 bars before
  build = mean vol of the 6 bars before the base > mean vol of the 24 bars before them
  surge = mean vol from base to t >= S * quiet mean vol
Leg end: extreme (low for down / high for up) is tracked after t and is CONFIRMED
when price comes back 1.5*ATR from it (no hindsight), or after 48 bars.
Retracement r = (price - extreme) / (base - extreme): 1.0 = back to the base (the yellow line).
Measured from the confirmation bar for 24/72/168 bars:
  reach_X     = touched level X at any time
  first_X     = touched level X BEFORE making a new extreme beyond the leg
"""
import numpy as np, pandas as pd, sys
src = sys.argv[1] if len(sys.argv) > 1 else 'data/btcusdt_1h.csv'
tf = sys.argv[2] if len(sys.argv) > 2 else '1h'
df = pd.read_csv(src, parse_dates=['time']).set_index('time')
if tf != '1h':
    df = df.resample(tf).agg(dict(o='first', h='max', l='min', c='last', v='sum')).dropna()
o, h, l, c, v = (df[k].values for k in 'ohlcv')
n = len(c)
pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
vmed = pd.Series(v).rolling(720, min_periods=200).median().shift(1).values
LV = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272]
HS = [24, 72, 168]
K, CONF, MAXLEG = 4.0, 1.5, 48

rows, t, last_end = [], 800, 0
while t < n - 200:
    found = None
    for d in (-1, 1):                                   # -1 = dump, +1 = pump
        win = c[t - 6:t]
        bi = t - 6 + (np.argmax(win) if d == -1 else np.argmin(win))
        base = c[bi]
        mv = (c[t] - base) * d
        if bi < 30 or bi <= last_end or np.isnan(atr[bi]) or mv < K * atr[bi]:
            continue
        q = v[bi - 24:bi].mean()
        b6 = v[bi - 6:bi].mean(); b24 = v[bi - 30:bi - 6].mean()
        found = dict(d=d, t=t, bi=bi, base=base, quiet=q / vmed[bi], build=b6 / b24,
                     surge=v[bi + 1:t + 1].mean() / q, size_atr=mv / atr[bi])
        break
    if not found:
        t += 1; continue
    d, base, a = found['d'], found['base'], atr[found['bi']]
    ext, ei = (l[t], t) if d == -1 else (h[t], t)
    j = t + 1
    while j < n and j - t <= MAXLEG:                    # extend the leg until confirmed
        if d == -1:
            if l[j] < ext: ext, ei = l[j], j
            elif h[j] - ext >= CONF * a: break
        else:
            if h[j] > ext: ext, ei = h[j], j
            elif ext - l[j] >= CONF * a: break
        j += 1
    leg = abs(base - ext)
    ev = dict(found, time=df.index[found['t']], ext=ext, leg_pct=leg / base, conf=j)
    for H in HS:
        seg_h, seg_l = h[j:j + H], l[j:j + H]
        if d == -1:
            retr = (np.maximum.accumulate(seg_h) - ext) / leg
            beyond = np.minimum.accumulate(seg_l) < ext
        else:
            retr = (ext - np.minimum.accumulate(seg_l)) / leg
            beyond = np.maximum.accumulate(seg_h) > ext
        # retracement already done between extreme and confirmation counts too
        pre = ((h[ei:j + 1].max() - ext) if d == -1 else (ext - l[ei:j + 1].min())) / leg
        retr = np.maximum(retr, pre)
        ev[f'max_{H}'] = retr.max()
        for L in LV:
            hit = np.argmax(retr >= L) if (retr >= L).any() else None
            ev[f'reach{L}_{H}'] = hit is not None
            nb = np.argmax(beyond) if beyond.any() else None
            ev[f'first{L}_{H}'] = hit is not None and (nb is None or hit <= nb)
        ev[f'newext_{H}'] = beyond.any()
    rows.append(ev)
    last_end = j; t = j + 1

E = pd.DataFrame(rows)
E.to_csv(f'retr/events_{tf}.csv', index=False)

def table(sub, name):
    if len(sub) < 15:
        print(f'\n{name}: only {len(sub)} events — too few'); return
    rng = np.random.default_rng(0)
    print(f'\n{name}  (n={len(sub)}, median leg {sub.leg_pct.median():.1%})')
    print(f"  {'level':>6} | {'reach 24':>9} {'reach 72':>9} {'reach 168':>10} | {'before new extreme (168)':>24}")
    for L in LV:
        cells = []
        for H in HS:
            x = sub[f'reach{L}_{H}'].values.astype(float)
            cells.append(f'{x.mean():>8.0%} ')
        x = sub[f'first{L}_168'].values.astype(float)
        bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
        cells.append(f'{x.mean():>12.0%} [{np.percentile(bs, 2.5):.0%}-{np.percentile(bs, 97.5):.0%}]')
        print(f'  {L:>6} | {cells[0]} {cells[1]} {cells[2]:>9} | {cells[3]:>24}')
    print(f"  median max retracement 24/72/168: " + ' / '.join(f"{sub[f'max_{H}'].median():.2f}" for H in HS)
          + f" | new extreme within 168: {sub['newext_168'].mean():.0%}")

E['pre'] = np.nan
VAR = {'strict (quiet<=0.8, build>1.1, surge>=3)': (E.quiet <= 0.8) & (E.build > 1.1) & (E.surge >= 3),
       'loose  (quiet<=1.0, build>1.0, surge>=2)': (E.quiet <= 1.0) & (E.build > 1.0) & (E.surge >= 2)}
pattern = VAR['loose  (quiet<=1.0, build>1.0, surge>=2)']
for d, nm in [(-1, 'DUMP'), (1, 'PUMP')]:
    S = E[E.d == d]
    print('\n' + '=' * 90 + f'\n{nm} — {tf} bars, impulse >= {K} ATR')
    table(S, f'[ALL {nm.lower()}s — control]')
    for vn, vm in VAR.items():
        table(S[vm[S.index]], f'[YOUR PATTERN {vn}]')
    table(S[~pattern[S.index]], f'[other {nm.lower()}s — not your pattern (loose)]')
    P = S[pattern[S.index]]
    if len(P) >= 30:
        for a_, b_ in [('2017', '2021-12-31'), ('2022', '2026-12-31')]:
            ts = P.time.astype(str)
            table(P[(ts >= a_) & (ts <= b_)], f'  pattern {a_}..{b_[:4]}')
