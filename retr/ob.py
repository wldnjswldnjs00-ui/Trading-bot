"""'First counter-candle at the extreme' retest (your 11/05 and 10/31 examples), 5m.

SHORT (after a pump)                          LONG (after a dump) = mirror
 1 pump: high - lowest low of last 24 bars >= K*ATR, and (VS=2) the volume of those
   24 bars >= 2x the quiet volume before them, or (VS=0) no volume condition on the move.
 2 top candle: the FIRST bearish 5m candle whose high is within 1 ATR of the
   running high, with volume >= VM x quiet volume (your 'first red candle at the
   top with volume'). VM=0 = no volume condition (control).  Its body = the zone: near edge = close, far edge = open.
   If price makes a new high before step 3, the top moves and we look again.
 3 pullback: price must first drop >= P*ATR below the body (the move away).
 4 retest: limit SHORT at the zone (near / mid / far edge of the body).
   stop  'sh'    : above the swing high (the wick) + 0.1 ATR   (resting stop)
         'tight' : above the body top + 0.5 ATR
   target 'pl'   : the pullback low seen before entry, or 1R / 2R / 3R
   be 1          : stop to entry after +1R
 No look-ahead: each step only uses closed bars. Same 5m bar touching stop and
 target => stop. Fees: entry maker 0.02%, target maker 0.02%, stop taker 0.05% (+SLIP).
usage: python retr/ob.py <5m csv> <TAG>"""
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
        fill = {}
        last_end = k
        for m in range(k + 1, min(k + 1 + 864, n)):
            for en, Z in zone.items():
                if en not in fill and ((d == 1 and h[m] >= Z * (1 + PEN)) or (d == -1 and l[m] <= Z * (1 - PEN))):
                    fill[en] = (m, PL)
            PL = min(PL, l[m]) if d == 1 else max(PL, h[m])
            if len(fill) == 3: break
        ev['retest'] = int('near' in fill)
        for en, st, tg, be in itertools.product(ENT, STP, TGT, BE):
            key = f'{en}|{st}|{tg}|be{be}'; R = None
            if en in fill:
                m0, pl = fill[en]; ent = zone[en]; stp = stops[st]
                risk = abs(stp - ent) / ent
                if (d == 1 and stp > ent) or (d == -1 and stp < ent):
                    T = pl if tg == 'pl' else ent + s * tg * risk * ent
                    cur = stp; moved = False
                    for mm in range(m0, min(m0 + 2592, n)):
                        if (s == -1 and h[mm] >= cur) or (s == 1 and l[mm] <= cur):
                            R = (s * (cur - ent) / ent - MAKER - TAKER) / risk; break
                        if mm > m0 and ((s == -1 and l[mm] <= T * (1 - PEN)) or (s == 1 and h[mm] >= T * (1 + PEN))):
                            R = (abs(T - ent) / ent - 2 * MAKER) / risk; break
                        if be and not moved and s * (c[mm] - ent) / ent >= risk: cur = ent; moved = True
                    else:
                        R = (s * (c[min(m0 + 2592, n - 1)] - ent) / ent - MAKER - TAKER) / risk
                    last_end = max(last_end, m0)
            ev[key] = R
        out.append(ev)
        t = max(k, last_end) + 1
    return out

rows = []
for K, VM, P, VS, d in itertools.product([4, 6, 8], [0, 1, 2, 3], [2, 4], [0, 2], [1, -1]):
    rows += run(K, VM, P, VS, d)
E = pd.DataFrame(rows); E['coin'] = TAG
E.to_csv(f'retr/ob_events_{TAG}.csv', index=False)  # PEN/SLIP set by env
print(TAG, len(E), 'setups;', E.groupby(['d']).size().to_dict())
