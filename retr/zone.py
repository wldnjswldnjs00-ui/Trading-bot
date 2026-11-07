"""5m: high-volume consolidation box (the yellow 'supply/demand zone') -> breakout impulse
-> does price come back to the box?  Dumps (short) and pumps (long).

Box  : B bars (12/24/36 = 1/2/3h) right before the breakout, range <= RMAX*ATR,
       mean volume >= VR x mean volume of the 48 bars before the box (volume woke up).
Break: first close outside the box. Impulse: price then travels >= K*ATR beyond the
       box edge within 36 bars (the 'dumped hard' part). Known at the bar it happens.
Leg end: extreme confirmed by a 1.5*ATR bounce (or 48 bars). All outcomes from there.
Return = high reaches the box LOW (dump) / low reaches the box HIGH (pump).
Trade: limit at the box edge, stop beyond the opposite box edge + 0.2*ATR,
       target T0 = the extreme, T50 = halfway back to the extreme.
       Cancel if the extreme breaks first or after 72h.  Fees: maker in 0.02%,
       maker target 0.02%, stop taker 0.05% + 0.02% slippage. Same bar => stop."""
import numpy as np, pandas as pd, sys
df = pd.read_csv('data/okx_btc_5m_trim.csv', parse_dates=['time']).set_index('time')
o, h, l, c, v = (df[k].values for k in 'ohlcv'); n = len(c)
pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
ema = pd.Series(c).ewm(span=600).mean().values
hour = df.index.hour.values
RMAX, VR, K, KBOX, CONF, HZ = 10.0, 1.5, 6.0, 2.0, 1.5, 864
MAKER, TAKER = 0.0002, 0.0007

rows, t, last = [], 400, 0
while t < n - HZ - 50:
    ev = None
    for B in (36, 24, 12):
        b0 = t - B
        if t <= last or b0 < 100: continue
        a = atr[b0]
        bh, bl = h[b0:t].max(), l[b0:t].min()
        if bh - bl > RMAX * a: continue
        vr = v[b0:t].mean() / max(v[b0 - 48:b0].mean(), 1e-9)
        if vr < VR: continue
        for d in (-1, 1):
            if (d == -1 and c[t] < bl) or (d == 1 and c[t] > bh):
                ev = dict(d=d, t=t, B=B, bh=bh, bl=bl, a=a, vr=vr); break
        if ev: break
    if not ev: t += 1; continue
    d, a, bh, bl = ev['d'], ev['a'], ev['bh'], ev['bl']
    edge = bl if d == -1 else bh
    # impulse must reach K*ATR beyond the edge within 36 bars
    seg = l[t:t + 37] if d == -1 else h[t:t + 37]
    far = (edge - seg) if d == -1 else (seg - edge)
    need = max(K * a, KBOX * (bh - bl))                  # dumped hard: vs ATR AND vs the box
    if far.max() < need: t += 1; continue
    kk = t + int(np.argmax(far >= need))                   # bar where the impulse became 'hard'
    ext = (l[t:kk + 1].min() if d == -1 else h[t:kk + 1].max())
    ei = t + int(np.argmin(l[t:kk + 1]) if d == -1 else np.argmax(h[t:kk + 1])); j = kk + 1
    while j < n and j - kk <= 72:
        if d == -1:
            if l[j] < ext: ext, ei = l[j], j
            elif h[j] - ext >= CONF * a: break
        else:
            if h[j] > ext: ext, ei = h[j], j
            elif ext - l[j] >= CONF * a: break
        j += 1
    if j >= n - HZ: break
    zone_edge, zone_far = (bl, bh) if d == -1 else (bh, bl)      # edge we return to, far side for the stop
    dist = abs(zone_edge - ext)
    pre = (h[ei:j + 1].max() if d == -1 else l[ei:j + 1].min())
    # outcomes
    sh, sl = h[j + 1:j + 1 + HZ], l[j + 1:j + 1 + HZ]
    back = (np.maximum.accumulate(np.r_[pre, sh]) >= zone_edge)[1:] if d == -1 else (np.minimum.accumulate(np.r_[pre, sl]) <= zone_edge)[1:]
    brk = (sl < ext) if d == -1 else (sh > ext)
    already = (pre >= zone_edge) if d == -1 else (pre <= zone_edge)
    bi_ = np.argmax(back) if back.any() else None
    ki_ = np.argmax(brk) if brk.any() else None
    f = dict(d=d, time=df.index[t], B=ev['B'], vr=ev['vr'], box_atr=(bh - bl) / a,
             imp_atr=dist / a, imp_box=dist / max(bh - bl, 1e-9), dist_pct=dist / ext,
             trend=int((c[t] - ema[t]) * d > 0), sess=['asia', 'europe', 'us', 'late'][min(hour[t] // 6, 3)],
             prelegs=int(((c[t - ev['B'] - 24] - c[t - ev['B']]) * -d) > 2 * a),   # a first leg into the box
             bounce=abs(pre - ext) / dist, already=int(already))
    for H, nm in [(288, '24h'), (864, '72h')]:
        f[f'back_{nm}'] = int(already or (bi_ is not None and bi_ < H))
    f['back_first'] = int(already or (bi_ is not None and (ki_ is None or bi_ <= ki_)))
    # trade
    for tg in ('T0', 'T50'):
        R = None
        if not already:
            side = d; entry = zone_edge; stop = zone_far - side * 0.2 * a
            tgt = ext if tg == 'T0' else (entry + ext) / 2
            risk = abs(stop - entry) / entry
            for k in range(j + 1, min(j + 1 + HZ, n)):
                if (side == -1 and l[k] < ext) or (side == 1 and h[k] > ext): break
                if (side == -1 and h[k] >= entry) or (side == 1 and l[k] <= entry):
                    for m in range(k, min(k + 3 * HZ, n)):
                        if (side == -1 and h[m] >= stop) or (side == 1 and l[m] <= stop):
                            R = (-risk - MAKER - TAKER) / risk; break
                        if m > k and ((side == -1 and l[m] <= tgt) or (side == 1 and h[m] >= tgt)):
                            R = (abs(entry - tgt) / entry - 2 * MAKER) / risk; break
                    else:
                        R = (side * (c[min(k + 3 * HZ, n - 1)] - entry) / entry - MAKER - TAKER) / risk
                    break
        f[f'R_{tg}'] = R
    rows.append(f); last = j; t = j + 1

E = pd.DataFrame(rows); E.to_csv('retr/zone_events_5m.csv', index=False)
ex = E[(E.time >= '2026-11-05 14:00') & (E.time <= '2026-11-05 15:00')]
print('YOUR EXAMPLE (2026-11-05 UTC):'); print(ex.round(2).to_string(index=False) if len(ex) else '  not detected')
split = E.time.iloc[int(len(E) * 0.6)]
E['half'] = np.where(E.time < split, 'train', 'test')
print(f'\n{len(E)} box-breakout impulses, {E.time.min():%Y-%m}..{E.time.max():%Y-%m}, train < {split:%Y-%m-%d}')
for d, nm in [(-1, 'DUMP from a box -> back up to the box low?  (SHORT there)'),
              (1, 'PUMP from a box -> back down to the box high? (LONG there)')]:
    S = E[E.d == d]
    print(f'\n===== {nm}  n={len(S)}')
    print(f"  back to the box: within 24h {S.back_24h.mean():.0%} | within 72h {S.back_72h.mean():.0%} "
          f"| before a new extreme {S.back_first.mean():.0%} | (median distance box->extreme {S.dist_pct.median():.1%}, {S.imp_box.median():.1f} box heights)")
    for tg in ('T0', 'T50'):
        col = f'R_{tg}'
        for hf in ('train', 'test'):
            x = S[S.half == hf][col].dropna()
            print(f'  trade {tg:<4}{hf:<6}: n={len(x):>4}  win {(x > 0).mean():>4.0%}  mean {x.mean():+.3f}R  total {x.sum():+.1f}R')
