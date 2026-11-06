"""v5: 1h wave-3 regime + 5m scalp entries in the same direction, futures sizing.

Owner's sizing rules:
- loss at stop incl. fees <= RISK of equity (max 3%)
- margin <= 40% of equity, leverage <= 20x  -> notional <= 8x equity
Size = RISK / (stop distance + round-trip fee), capped at 8x equity.
Fees on notional: maker 0.02% or taker 0.05% per side (Binance USDT-M base tier).
"""
import sys, bisect, itertools, numpy as np, pandas as pd
from engine import atr, zigzag
from confluence import fib_hit, vp_strength

MAXNOTIONAL = 0.40 * 20
SPLIT = pd.Timestamp('2023-01-01', tz='UTC'); BUF = 0.0005


def load(p):
    d = pd.read_csv(p); d['time'] = pd.to_datetime(d.time, format='ISO8601', utc=True); return d


H = load('data/btcusdt_1h.csv'); M = load('data/btcusdt_5m.csv')
Ah, Am = atr(H), atr(M)


def wave2_entries(df, A, K, ratio, zone, maxstop, vpwin, need_conf):
    """Wave-3 starts: big-degree wave 1, entry on first lower-degree swing in the
    wave-2 zone. Returns (entry_bar, up, entry, stop, target_ext, conf)."""
    o, h, l, v = df.o.values, df.h.values, df.l.values, df.v.values
    big = zigzag(df, K, A); low = zigzag(df, K / ratio, A); lk = [q[3] for q in low]; out = []
    for j in range(1, len(big)):
        p0, p1 = big[j - 1], big[j]; up = p1[2] == 1; s = 1 if up else -1
        w1 = s * (p1[1] - p0[1])
        if w1 <= 0: continue
        nxt = big[j + 1][3] if j + 1 < len(big) else len(o)
        for q in low[bisect.bisect_right(lk, p1[3]):]:
            if q[3] >= nxt: break
            if q[2] != (-1 if up else 1) or q[0] <= p1[0]: continue
            if s * (q[1] - p0[1]) <= 0: break
            r = s * (p1[1] - q[1]) / w1
            if not (zone[0] <= r < zone[1]): continue
            e = q[3] + 1
            if e >= len(o): break
            entry = o[e]; stop = q[1] * (1 - s * BUF); dist = s * (entry - stop) / entry
            if dist <= 0 or dist > maxstop: continue
            conf = fib_hit(q[1], p0[1], p1[1], 0.003) + (vp_strength(h, l, v, max(0, p1[0] - vpwin), p1[0], q[1], 0.003) >= 1.5)
            if need_conf and conf < 2: continue
            out.append((e, up, entry, stop, q[1] + s * 1.618 * w1, conf))
            break
    return out


def regimes(K):
    """1h wave-3 windows: from 1h entry until its stop or 1.618 target is hit."""
    o, h, l = H.o.values, H.h.values, H.l.values; reg = []
    for e, up, entry, stop, tgt, conf in wave2_entries(H, Ah, K, 3, (0.382, 0.886), 0.02, 720, False):
        for t in range(e, len(o)):
            if (l[t] <= stop or h[t] >= tgt) if up else (h[t] >= stop or l[t] <= tgt): break
        reg.append((H.time[e], H.time[t], up))
    return reg


_E, _R = {}, {}


def run(Kh, Km, maxstop, tgtR, fee, tstop, risk, use_regime, need_conf):
    o, h, l, c = M.o.values, M.h.values, M.l.values, M.c.values
    if Km not in _E: _E[Km] = wave2_entries(M, Am, Km, 3, (0.382, 0.886), 0.01, 2016, False)
    if use_regime and Kh not in _R: _R[Kh] = regimes(Kh)
    reg = _R[Kh] if use_regime else None
    rs = [r[0] for r in reg] if reg else None
    tr = []; busy = -1
    for e, up, entry, stop, _, conf in _E[Km]:
        if e <= busy: continue
        if need_conf and conf < 2: continue
        if (entry - stop) / entry * (1 if up else -1) > maxstop: continue
        if reg:
            i = bisect.bisect_right(rs, M.time[e]) - 1
            if i < 0 or not (reg[i][0] <= M.time[e] <= reg[i][1]) or reg[i][2] != up: continue
        s = 1 if up else -1; dist = s * (entry - stop) / entry
        cost = 2 * fee                                    # round trip, fraction of notional
        notional = min(risk / (dist + cost), MAXNOTIONAL)
        T = entry + s * tgtR * dist * entry
        end = min(len(o), e + tstop); px = None
        for t in range(e, end):
            if (l[t] <= stop) if up else (h[t] >= stop): px = stop; break
            if (h[t] >= T) if up else (l[t] <= T): px = T; break
        if px is None: t = end - 1; px = c[t]
        pnl = notional * (s * (px - entry) / entry - cost)   # fraction of equity
        tr.append((M.time[e], pnl)); busy = t
    return pd.DataFrame(tr, columns=['time', 'pnl'])


def summary(T):
    if len(T) < 10: return dict(n=len(T))
    eq = np.cumprod(1 + T.pnl.values); eq = np.maximum(eq, 0)
    m = pd.Series(1 + T.pnl.values, index=T.time).groupby(T.time.dt.to_period('M').values).prod() - 1
    days = (T.time.iloc[-1] - T.time.iloc[0]).days or 1
    return dict(n=len(T), per_day=len(T) / days, win=(T.pnl > 0).mean(), avg=T.pnl.mean(),
                mdd=(eq / np.maximum.accumulate(eq) - 1).min(), med_month=m.median(),
                best_month=m.max(), worst_month=m.min(), ruined=bool(eq.min() <= 0.05))


if __name__ == '__main__':
    rows = []
    for Kh, Km, ms, tg, fee, ts, reg, nc in itertools.product(
            [6], [4, 6], [0.003, 0.005, 0.008], [1.5, 2.0, 3.0], [0.0002, 0.0005], [48], [True, False], [False, True]):
        T = run(Kh, Km, ms, tg, fee, ts, 0.01, reg, nc)          # 1% risk to find the edge first
        if T.empty: continue
        a = summary(T[T.time < SPLIT]); b = summary(T[T.time >= SPLIT])
        rows.append(dict(Km=Km, maxstop=ms, tgtR=tg, fee=fee, regime=reg, conf=nc,
                         **{f'{k}_in': v for k, v in a.items()}, **{f'{k}_out': v for k, v in b.items()}))
    R = pd.DataFrame(rows); R.to_csv('grid5.csv', index=False)
    pd.set_option('display.width', 300)
    cols = ['Km', 'maxstop', 'tgtR', 'fee', 'regime', 'conf', 'n_in', 'win_in', 'avg_in', 'n_out', 'per_day_out', 'win_out', 'avg_out', 'mdd_out']
    print(R.sort_values('avg_out', ascending=False)[cols].head(15).round(4).to_string(index=False))
    print('positive both:', ((R.avg_in > 0) & (R.avg_out > 0)).sum(), 'of', len(R))
