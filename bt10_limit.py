"""ICT with resting limit orders, conservative fills.

A limit at the OTE level only fills when price trades THROUGH it by THRU
(a touch is not a fill — queue position). Entry and take-profit pay maker,
the stop pays taker. Reports how many orders were placed vs filled.
Same structure, stop (prior swing) and 10%-of-margin sizing as bt8_ict.
"""
import itertools, numpy as np, pandas as pd
import bt8_ict as I
from engine import atr, zigzag

THRU = 0.0002
MAKER, TAKER = 0.00014, 0.00035          # OKX 0.02% / 0.05% after ~30% rebate


def orders(df, K, level):
    A = atr(df); piv = zigzag(df, K, A); o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    placed = 0; S = []
    for i in range(3, len(piv)):
        p1, p2, p3, p4 = piv[i - 3], piv[i - 2], piv[i - 1], piv[i]
        up = p4[2] == 1; s = 1 if up else -1
        if not (s * (p3[1] - p1[1]) > 0 and s * (p4[1] - p2[1]) > 0): continue
        leg = s * (p4[1] - p3[1])
        if leg <= 0: continue
        lvl = p4[1] - s * level * leg; inv = p4[1] - s * 0.79 * leg
        start = p4[3] + 1; end = piv[i + 1][3] if i + 1 < len(piv) else len(o)
        if start >= len(o) or ((o[start] <= lvl) if up else (o[start] >= lvl)): continue   # already past: no resting order
        placed += 1
        for t in range(start, min(end + 1, len(o))):
            if (h[t] > p4[1]) if up else (l[t] < p4[1]): break
            if (l[t] < lvl * (1 - THRU)) if up else (h[t] > lvl * (1 + THRU)):
                S.append(dict(e=t, up=up, entry=lvl, stop=p3[1] * (1 - s * I.BUF), tp1=p4[1], tp2=p4[1] + s * 0.62 * leg))
                break
            if (c[t] < inv) if up else (c[t] > inv): break
    return S, placed


def run(df, S, bias, mode):
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    rows = []; busy = -1
    for x in S:
        e = x['e']; s = 1 if x['up'] else -1
        if e <= busy or (bias is not None and bias[e] != s): continue
        entry, stop = x['entry'], x['stop']; d = s * (entry - stop) / entry
        if d <= 0 or d > 0.10: continue
        unit = d + MAKER + TAKER; pos = 1.0; pnl = 0.0; half = False; t = e
        while t < len(o):
            if (l[t] <= stop) if s > 0 else (h[t] >= stop):          # fill bar included: worst case
                pnl += pos * (s * (stop - entry) / entry - MAKER - TAKER); pos = 0; break
            if t > e and mode == 'half' and not half and ((h[t] > x['tp1']) if s > 0 else (l[t] < x['tp1'])):
                pnl += 0.5 * (s * (x['tp1'] - entry) / entry - 2 * MAKER); pos = 0.5; half = True
                stop = entry * (1 + s * (MAKER + TAKER)); t += 1; continue
            tgt = x['tp1'] if mode == 'full' else x['tp2']
            if t > e and ((h[t] > tgt) if s > 0 else (l[t] < tgt)):    # TP limit also needs a trade-through
                pnl += pos * (s * (tgt - entry) / entry - 2 * MAKER); pos = 0; break
            t += 1
        if pos > 0: t = len(o) - 1; pnl += pos * (s * (c[t] - entry) / entry - MAKER - TAKER)
        rows.append((df.time[e], pnl / unit, d)); busy = t
    return pd.DataFrame(rows, columns=['time', 'R', 'd'])


if __name__ == '__main__':
    rows = []
    for ltf, htf in [('15min', '4h'), ('30min', '12h'), ('5min', '1h')]:
        df = I.bars(ltf); bias = I.htf_bias(htf, 3, df.time)
        for K, level in itertools.product((3, 4, 6), (0.5, 0.618, 0.705)):
            S, placed = orders(df, K, level)
            for use_bias, mode in itertools.product((False, True), ('full', 'half')):
                T = run(df, S, bias if use_bias else None, mode)
                a = I.summ(T[T.time < I.SPLIT]); b = I.summ(T[T.time >= I.SPLIT])
                rows.append(dict(ltf=ltf, htf=htf if use_bias else '-', K=K, level=level, mode=mode,
                                 fill_rate=len(S) / max(placed, 1),
                                 **{f'{k}_in': v for k, v in a.items()}, **{f'{k}_out': v for k, v in b.items()}))
        print(ltf, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv('grid10_limit.csv', index=False)
    pd.set_option('display.width', 320)
    ok = R[(R.n_in >= 50) & (R.n_out >= 30)].copy(); ok['m'] = ok[['avgR_in', 'avgR_out']].min(axis=1)
    print('configs', len(R), 'positive both', (ok.m > 0).sum(), 'median out', ok.avgR_out.median().round(3))
    cols = ['ltf', 'htf', 'K', 'level', 'mode', 'fill_rate', 'n_in', 'win_in', 'avgR_in', 'n_out', 'per_day_out', 'win_out',
            'avgR_out', 'R_month_out', 'mdd_out', 'med_month_out', 'med_stop_out']
    print(ok.sort_values('m', ascending=False)[cols].head(15).round(3).to_string(index=False))
