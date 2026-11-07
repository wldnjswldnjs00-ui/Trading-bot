"""Volume box -> impulse -> return to the box: every sub-variant, honestly scored.

DIMENSIONS
 zone   edge : box low (dump) / box high (pump)
        mid  : box middle
        vwap : volume-weighted price of the box
        poc  : price of the highest-volume bar in the box (where most traded)
 entry  L    : resting limit at the zone
        LV   : limit, but pulled if the bounce into the zone is NOT weak
               (bounce volume / impulse volume must be < 0.5, checked on closed bars)
        RJ   : touch the zone, then wait for a 5m close back out of it (rejection),
               enter at that close (taker). Stop beyond the touch high/low.
        RJV  : RJ + the bounce volume weak (< 0.5 of impulse)
 stop   L/LV : far side of the box + 0.2 ATR      RJ/RJV : wick of the touch + 0.2 ATR
 target ext (the impulse extreme), 1R, 2R, 3R
 be     0/1  : move stop to entry once price has moved +1R
FILTERS (known before entry): impulse size vs ATR, first-bounce strength, trend.
Fees: maker 0.02%, taker 0.05% + 0.02% slippage. Same-bar stop & target => stop.
SCORING: train = first 60% of events, test = rest; a variant 'survives' only if
mean R > 0 in train, test, and both halves of test with >= 30 test trades.
The number of survivors is compared with what pure luck would give."""
import numpy as np, pandas as pd, itertools, sys, warnings
warnings.simplefilter('ignore')
SRC = sys.argv[1] if len(sys.argv) > 1 else 'data/okx_btc_5m_trim.csv'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'BTC'
df = pd.read_csv(SRC, parse_dates=['time']).set_index('time')
df = df[df.index >= '2020-07-01']
h, l, c, v = (df[k].values for k in 'hlcv'); n = len(c)
pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
ema = pd.Series(c).ewm(span=600).mean().values
RMAX, VR, K, KBOX, CONF, HZ = 10.0, 1.5, 6.0, 2.0, 1.5, 864
MAKER, TAKER = 0.0002, 0.0007

# ---------- detect events (same rules as zone.py) ----------
ev, t, last = [], 400, 0
while t < n - 4 * HZ:
    got = None
    for B in (36, 24, 12):
        b0 = t - B
        if t <= last or b0 < 100: continue
        a = atr[b0]; bh, bl = h[b0:t].max(), l[b0:t].min()
        if bh - bl > RMAX * a: continue
        vr = v[b0:t].mean() / max(v[b0 - 48:b0].mean(), 1e-9)
        if vr < VR: continue
        for d in (-1, 1):
            if (d == -1 and c[t] < bl) or (d == 1 and c[t] > bh): got = (d, B, b0, a, bh, bl, vr); break
        if got: break
    if not got: t += 1; continue
    d, B, b0, a, bh, bl, vr = got
    edge = bl if d == -1 else bh
    seg = l[t:t + 37] if d == -1 else h[t:t + 37]
    far = (edge - seg) if d == -1 else (seg - edge)
    need = max(K * a, KBOX * (bh - bl))
    if far.max() < need: t += 1; continue
    kk = t + int(np.argmax(far >= need))
    ei = t + int(np.argmin(l[t:kk + 1]) if d == -1 else np.argmax(h[t:kk + 1]))
    ext = l[ei] if d == -1 else h[ei]; j = kk + 1
    while j < n and j - kk <= 72:
        if d == -1:
            if l[j] < ext: ext, ei = l[j], j
            elif h[j] - ext >= CONF * a: break
        else:
            if h[j] > ext: ext, ei = h[j], j
            elif ext - l[j] >= CONF * a: break
        j += 1
    tp = (h[b0:t] + l[b0:t] + c[b0:t]) / 3
    zones = dict(edge=edge, mid=(bh + bl) / 2, vwap=(tp * v[b0:t]).sum() / max(v[b0:t].sum(), 1e-9),
                 poc=tp[np.argmax(v[b0:t])])
    pre = h[ei:j + 1].max() if d == -1 else l[ei:j + 1].min()
    ev.append(dict(d=d, t=t, j=j, ei=ei, ext=ext, a=a, bh=bh, bl=bl, time=df.index[t], vr=vr,
                   imp_v=v[t:ei + 1].mean(), imp_atr=abs(edge - ext) / a,
                   bounce=abs(pre - ext) / max(abs(edge - ext), 1e-9),
                   trend=int((c[t] - ema[t]) * d > 0), **{f'z_{k}': z for k, z in zones.items()}))
    last = j; t = j + 1
E = pd.DataFrame(ev)

def sim(e, zone, entry, tgt, be):
    d, j, ext, a = e['d'], int(e['j']), e['ext'], e['a']
    side = d; Z = e[f'z_{zone}']
    far_side = e['bh'] if d == -1 else e['bl']
    if (side == -1 and Z <= ext) or (side == 1 and Z >= ext): return None
    # the bounce already went through the zone before we could act -> skip
    if (side == -1 and h[int(e['ei']):j + 1].max() >= Z) or (side == 1 and l[int(e['ei']):j + 1].min() <= Z): return None
    touch_hi = None
    for k in range(j + 1, min(j + 1 + HZ, n)):
        if (side == -1 and l[k] < ext) or (side == 1 and h[k] > ext): return None       # new extreme first
        if entry in ('LV', 'RJV'):
            bv = v[j:k].mean() / max(e['imp_v'], 1e-9)                                      # closed bars only
        touched = (h[k] >= Z) if side == -1 else (l[k] <= Z)
        if entry in ('L', 'LV'):
            if not touched: continue
            if entry == 'LV' and bv >= 0.5: return None
            ent = Z; stop = far_side - side * 0.2 * a; fee_in = MAKER; k0 = k
            break
        else:
            if touch_hi is None:
                if touched:
                    if entry == 'RJV' and bv >= 0.5: return None
                    touch_hi, tk = (h[k] if side == -1 else l[k]), k
                continue
            touch_hi = max(touch_hi, h[k]) if side == -1 else min(touch_hi, l[k])
            if k - tk > 12: return None
            if (side == -1 and c[k] < Z) or (side == 1 and c[k] > Z):
                ent = c[k]; stop = touch_hi - side * 0.2 * a; fee_in = TAKER; k0 = k + 1
                break
    else:
        return None
    risk = abs(stop - ent) / ent
    if risk <= 0: return None
    T = ext if tgt == 'ext' else ent + side * tgt * risk * ent
    st = stop; moved = False
    for m in range(k0, min(k0 + 3 * HZ, n)):
        if (side == -1 and h[m] >= st) or (side == 1 and l[m] <= st):
            pnl = side * (st - ent) / ent
            return (pnl - fee_in - TAKER) / risk
        if m > k0 - (entry in ('L', 'LV')) and ((side == -1 and l[m] <= T) or (side == 1 and h[m] >= T)):
            if m == k0 and entry in ('L', 'LV'):
                pass
            else:
                return (abs(T - ent) / ent - fee_in - MAKER) / risk
        if be and not moved and side * (c[m] - ent) / ent >= risk:
            st = ent; moved = True
    m = min(k0 + 3 * HZ, n - 1)
    return (side * (c[m] - ent) / ent - fee_in - TAKER) / risk

V = list(itertools.product(['edge', 'mid', 'vwap', 'poc'], ['L', 'LV', 'RJ', 'RJV'], ['ext', 1, 2, 3], [0, 1]))
cols = {f'{zone}|{entry}|{tgt}|be{be}': [sim(e, zone, entry, tgt, be) for _, e in E.iterrows()] for zone, entry, tgt, be in V}
E = pd.concat([E, pd.DataFrame(cols)], axis=1); E['coin'] = TAG
E.to_csv(f'retr/zone2_events_{TAG}.csv', index=False)
print(f'{len(E)} events, {len(V)} trade variants')
