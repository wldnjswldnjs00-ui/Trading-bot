"""5m: after a volume-surge impulse, when does price RETRACE and then BREAK the
low (dump) / high (pump) of the impulse?  Find the conditions, then trade it.

Event (causal): move >= K*ATR within 6 bars with a volume surge (>= 2x quiet vol).
Leg end confirmed when price bounces 1.5*ATR off the extreme (or 48 bars).
Everything below is measured from that confirmation bar (no hindsight).

Features known at confirmation:
  size     leg / ATR                     surge   impulse vol / quiet vol
  quiet    quiet vol / 60h median vol    dur     bars from start to extreme
  bvol     bounce vol / impulse vol      trend   price on the move's side of 50h EMA
  slope    50h EMA slope in the move's direction
  sweep    impulse broke the prior 24h low (dump) / high (pump)
  sess     UTC hour bucket               legpct  leg size in %
Outcome (24h = 288 bars): did it break the extreme? how deep was the retrace first?
Trade: limit at retrace E of the leg, stop at retrace S, target = the extreme
(tick beyond it). Cancel if the extreme breaks before fill. Same bar stop+target
=> stop. Fees: maker 0.02% in, maker 0.02% target, taker 0.05%+0.02% slip stop."""
import numpy as np, pandas as pd, sys, itertools
src = sys.argv[1] if len(sys.argv) > 1 else 'data/okx_btc_5m_trim.csv'
df = pd.read_csv(src, parse_dates=['time']).set_index('time')
h, l, c, v = (df[k].values for k in 'hlcv')
n = len(c); pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
vmed = pd.Series(v).rolling(720, min_periods=200).median().shift(1).values
ema = pd.Series(c).ewm(span=600).mean().values
lo288 = pd.Series(l).rolling(288).min().shift(1).values
hi288 = pd.Series(h).rolling(288).max().shift(1).values
hour = df.index.hour.values
K, M, CONF, MAXLEG, HZ = 3.0, 2.0, 1.5, 48, 288
MAKER, TAKER = 0.0002, 0.0007

rows, t, last = [], 3000, 0
while t < n - HZ - 5:
    ev = None
    for d in (-1, 1):
        win = c[t - 6:t]
        bi = t - 6 + (np.argmax(win) if d == -1 else np.argmin(win))
        if bi <= last or (c[t] - c[bi]) * d < K * atr[bi]: continue
        w0 = max(bi - 12, last + 1)
        q = v[w0 - 24:w0].mean()
        idx = [k for k in range(w0, t + 1) if v[k] >= M * q]
        if not idx or q <= 0: continue
        ev = dict(d=d, t=t, bi=bi, s=idx[0], q=q); break
    if not ev: t += 1; continue
    d, bi, s, a = ev['d'], ev['bi'], ev['s'], atr[ev['bi']]
    start = c[bi]
    ext, ei, j = (l[t], t, t + 1) if d == -1 else (h[t], t, t + 1)
    while j < n and j - t <= MAXLEG:
        if d == -1:
            if l[j] < ext: ext, ei = l[j], j
            elif h[j] - ext >= CONF * a: break
        else:
            if h[j] > ext: ext, ei = h[j], j
            elif ext - l[j] >= CONF * a: break
        j += 1
    if j >= n - HZ: break
    leg = abs(start - ext)
    imp_v = v[bi + 1:ei + 1].mean() if ei > bi else v[bi]
    f = dict(d=d, time=df.index[j], j=j, ext=ext, start=start, leg=leg,
             size=leg / a, surge=imp_v / ev['q'], quiet=ev['q'] / vmed[bi],
             dur=ei - bi, bvol=v[ei + 1:j + 1].mean() / imp_v if j > ei else np.nan,
             trend=int((c[bi] - ema[bi]) * d > 0), slope=int((ema[bi] - ema[bi - 72]) * d > 0),
             sweep=int((ext < lo288[bi]) if d == -1 else (ext > hi288[bi])),
             sess=['asia', 'europe', 'us', 'late'][min(hour[j] // 6, 3)], legpct=leg / start,
             conf_retr=((h[ei:j + 1].max() - ext) if d == -1 else (ext - l[ei:j + 1].min())) / leg)
    # outcome
    seg_h, seg_l = h[j + 1:j + 1 + HZ], l[j + 1:j + 1 + HZ]
    brk = (seg_l < ext) if d == -1 else (seg_h > ext)
    bk = np.argmax(brk) if brk.any() else HZ
    pre_h = np.r_[h[ei:j + 1], seg_h[:bk]]; pre_l = np.r_[l[ei:j + 1], seg_l[:bk]]
    f['brk'] = int(brk.any()); f['brk_bars'] = bk
    f['retr_before'] = ((pre_h.max() - ext) if d == -1 else (ext - pre_l.min())) / leg
    rows.append(f); last = j; t = j + 1

E = pd.DataFrame(rows)

def trade(f, E_, S_):
    d, ext, leg, j = f['d'], f['ext'], f['leg'], int(f['j'])
    side = d
    entry = ext - side * E_ * leg            # dump: ext + E*leg (short above the low)
    stop = ext - side * S_ * leg
    tgt = ext + side * 1e-9
    if (side == -1 and c[j] >= entry) or (side == 1 and c[j] <= entry): return None
    risk = abs(stop - entry) / entry
    for k in range(j + 1, min(j + 1 + HZ, n)):
        if (side == -1 and l[k] < ext) or (side == 1 and h[k] > ext): return None
        if (side == -1 and h[k] >= entry) or (side == 1 and l[k] <= entry):
            for m in range(k, min(k + 3 * HZ, n)):
                if (side == -1 and h[m] >= stop) or (side == 1 and l[m] <= stop):
                    return (-risk - MAKER - TAKER) / risk
                if m > k and ((side == -1 and l[m] <= tgt) or (side == 1 and h[m] >= tgt)):
                    return (abs(entry - ext) / entry - 2 * MAKER) / risk
            return (side * (c[min(k + 3 * HZ, n - 1)] - entry) / entry - MAKER - TAKER) / risk  # time exit
    return None

ES = [(0.382, 0.786), (0.382, 1.0), (0.5, 0.786), (0.5, 1.0), (0.618, 1.0), (0.618, 1.272), (0.786, 1.272)]
for e_, s_ in ES:
    E[f'R_{e_}_{s_}'] = [trade(f, e_, s_) for _, f in E.iterrows()]
E.to_csv('retr/cont_events_5m.csv', index=False)
split = E.time.iloc[int(len(E) * 0.6)]
E['half'] = np.where(E.time < split, 'train', 'test')
print(f'{len(E)} impulses on 5m ({E.time.min():%Y-%m}..{E.time.max():%Y-%m}), train < {split:%Y-%m-%d}')
for d, nm in [(-1, 'DUMP (short the bounce)'), (1, 'PUMP (long the dip)')]:
    S = E[E.d == d]
    print(f'\n===== {nm}: n={len(S)} | breaks the extreme within 24h: {S.brk.mean():.0%} '
          f'| median retrace before the break: {S[S.brk == 1].retr_before.median():.2f}')
    q = S[S.brk == 1].retr_before.quantile([.25, .5, .75, .9]).round(2).to_dict()
    print(f'   retrace-before-break quartiles (25/50/75/90%): {q}')
    print('   trade (entry retr, stop retr) -> mean R train | test | n test | win test')
    for e_, s_ in ES:
        col = f'R_{e_}_{s_}'
        a_, b_ = S[S.half == 'train'][col].dropna(), S[S.half == 'test'][col].dropna()
        print(f'     {e_:<5} {s_:<5}  {a_.mean():+.3f} | {b_.mean():+.3f} | {len(b_):>5} | {(b_ > 0).mean():.0%}')
