"""Averaging-down version: 10% of seed x 10x per entry (= 1x seed notional), one add of the same size.
YOUR version: enter at the signal (market), stop right next to entry, big target,
40% of seed as margin x 35x leverage = position 14x the seed. Same setups as ob.py
(K=4, no volume filter, pullback 4 ATR), both sides. Same-bar stop+target => stop."""
import numpy as np, pandas as pd, itertools, sys, os
SRC = sys.argv[1]; TAG = sys.argv[2]
SLIP = float(os.environ.get('SLIP', '0.0001'))
PEN = float(os.environ.get('PEN', '0.0002'))      # limit orders fill only if price trades THROUGH by PEN
df = pd.read_csv(SRC, parse_dates=['time']).set_index('time'); df = df[df.index >= '2020-07-01']
o, h, l, c, v = (df[k].values.astype(float) for k in 'ohlcv'); n = len(c)
pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
lo24 = pd.Series(l).rolling(24).min().values; hi24 = pd.Series(h).rolling(24).max().values
v24 = pd.Series(v).rolling(24).mean().values
quiet = pd.Series(v).shift(24).rolling(48).mean().values
ema = pd.Series(c).ewm(span=600).mean().values
MAKER, TAKER = 0.0002, 0.0005 + SLIP
ENT, STP, TGT, BE = ['near', 'mid', 'far'], ['sh', 'tight'], ['pl', 1, 2, 3], [0, 1]
TRACK = os.environ.get('TRACK')            # e.g. 'near|sh|pl|be0': also store entry/exit times

def run(K, VM, P, VS, d):
    """d = +1: pump -> short.  d = -1: dump -> long. Returns list of event dicts."""
    out, t = [], 200
    s = -d                                       # trade side
    while t < n - 1000:
        a = atr[t]
        move = (h[t] - lo24[t]) if d == 1 else (hi24[t] - l[t])
        if not (move >= K * a and v24[t] >= VS * quiet[t] and quiet[t] > 0): t += 1; continue
        q = quiet[t]; SH = h[t] if d == 1 else l[t]
        k, ob, m = t, None, t
        SH_ob = None
        while k < min(t + 72, n - 1):            # find the top candle, then the pullback
            if ob is None:
                SH = max(SH, h[k]) if d == 1 else min(SH, l[k])
                counter = (c[k] < o[k]) if d == 1 else (c[k] > o[k])
                near_top = (h[k] >= SH - atr[k]) if d == 1 else (l[k] <= SH + atr[k])
                if counter and near_top and v[k] >= VM * q: ob = k; SH_ob = SH
                k += 1; continue
            if (d == 1 and h[k] > SH_ob) or (d == -1 and l[k] < SH_ob):   # new extreme -> top moves
                ob = None; continue
            if (d == 1 and l[k] <= c[ob] - P * atr[ob]) or (d == -1 and h[k] >= c[ob] + P * atr[ob]):
                break                             # pullback done at bar k
            k += 1
            continue
        else:
            t = k + 1; continue
        if ob is None: t = k + 1; continue
        # ---- the setup exists at bar k (known at its close)
        a = atr[ob]; SHv = SH_ob
        near, far = c[ob], o[ob]
        zone = dict(near=near, mid=(near + far) / 2, far=far)
        stops = dict(sh=SHv + d * 0.1 * a, tight=far + d * 0.5 * a)
        ev = dict(d=d, time=df.index[ob], K=K, VM=VM, P=P, VS=VS, volx=v[ob] / q,
                  wick=((h[ob] - max(o[ob], c[ob])) if d == 1 else (min(o[ob], c[ob]) - l[ob])) / max(h[ob] - l[ob], 1e-9),
                  body_atr=abs(o[ob] - c[ob]) / a, move_atr=move / atr[t],
                  trend=int((c[ob] - ema[ob]) * d > 0))
        PL = l[ob:k + 1].min() if d == 1 else h[ob:k + 1].max()
        Z = zone['near']; m0 = None
        for m in range(k + 1, min(k + 1 + 864, n)):
            if (d == 1 and h[m] >= Z) or (d == -1 and l[m] <= Z): m0 = m; break
            PL = min(PL, l[m]) if d == 1 else max(PL, h[m])
        if m0 is None: t = k + 1; continue
        ent = Z * (1 + s * SLIPM)                                  # market order at the signal, pays slippage (adverse)
        ev['t_in'] = df.index[m0]; ev['pl_dist'] = abs(PL - ent) / ent
        for A, S, T in itertools.product(ADDS, STOPS2, TGTS2):
            e1 = ent; e2 = e1 * (1 - s * A) if A else None
            stp = (e2 if A else e1) * (1 - s * S)
            avg, size, fee = e1, 1, TAKER_IN                     # first entry: market at the signal
            tgt = avg * (1 + s * T); res = None
            for mm in range(m0, min(m0 + 2592, n)):
                adv = l[mm] if s == 1 else h[mm]; fav = h[mm] if s == 1 else l[mm]
                added_now = False
                if A and size == 1 and ((s == 1 and adv <= e2) or (s == -1 and adv >= e2)):
                    size = 2; avg = (e1 + e2) / 2; fee += MAKER; tgt = avg * (1 + s * T); added_now = True   # averaging down (limit)
                if (s == 1 and adv <= stp) or (s == -1 and adv >= stp):
                    pnl = s * (stp - e1) / e1 + (s * (stp - e2) / e2 if size == 2 else 0)
                    res = (pnl - fee - size * TAKER_OUT, 'stop', mm, size); break
                if mm > m0 and not added_now and ((s == 1 and fav >= tgt) or (s == -1 and fav <= tgt)):
                    pnl = s * (tgt - e1) / e1 + (s * (tgt - e2) / e2 if size == 2 else 0)
                    res = (pnl - fee - size * MAKER, 'tgt', mm, size); break
            else:
                mm = min(m0 + 2592, n - 1)
                pnl = s * (c[mm] - e1) / e1 + (s * (c[mm] - e2) / e2 if size == 2 else 0)
                res = (pnl - fee - size * TAKER_OUT, 'time', mm, size)
            key = f'{A}|{S}|{T}'
            ev[key] = res[0]                                      # P&L in units of ONE entry's notional
            ev[key + '|out'] = df.index[res[2]]
            ev[key + '|added'] = int(res[3] == 2)
        out.append(ev)
        t = m0 + 1
    return out

GRID = os.environ.get('GRID')               # e.g. '4,0,4,0,-1' to run one setting only

ADDS = [0, 0.003, 0.005, 0.01]          # 0 = no averaging; else add at this adverse move
STOPS2 = [0.003, 0.005, 0.01]           # stop distance beyond the LAST entry
TGTS2 = [0.005, 0.01, 0.02, 0.03]       # target measured from the average price
SLIPM = float(os.environ.get('SLIPM', '0.0002'))    # market-order slippage at entry
TAKER_IN = 0.0005 + 0.0                              # slippage already in the entry price
TAKER_OUT = 0.0005 + float(os.environ.get('SLIPS', '0.0002'))  # stop: taker + slippage
rows = []
for d in (1, -1):
    rows += run(4, 0, 4, 0, d)
E = pd.DataFrame(rows); E['coin'] = TAG
E.to_csv(f'retr/avg_{TAG}.csv', index=False); print(TAG, len(E))
