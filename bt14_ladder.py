"""ICT 30m ladder: equal limit orders at several discount levels, 30 coins.

Size so that if every rung fills and the prior-swing stop hits, loss = 1R
(fees included). Partial fills lose less. Unfilled rungs are cancelled at the
first take-profit, a new leg extreme, or a candle body closing beyond 0.79
(rungs at/above 0.79 cancel only on a close beyond the swing stop side).
Half of the filled size off at the prior high, stop to average entry, rest to
the -0.62 extension. Optional order-flow filters from bt13 (A: impulse delta,
C: pullback delta up to the first touch).
"""
import glob, os, itertools, numpy as np, pandas as pd
from engine import atr, zigzag
from bt13_ict_cvd import load30, MAKER, TAKER, THRU, BUF, SPLIT

LADDERS = {'0.5': (0.5,), '0.618': (0.618,), '0.705': (0.705,), '0.79': (0.79,),
           '0.5-0.705': (0.5, 0.618, 0.705), '0.618-0.79': (0.618, 0.705, 0.79), '0.5-0.79': (0.5, 0.618, 0.705, 0.79)}
FILTERS = [('none', -9, -9), ('A0', 0.0, -9), ('C0', -9, 0.0), ('A0+C0', 0.0, 0.0)]


def run_sym(df, K):
    A = atr(df); piv = zigzag(df, K, A)
    o, h, l, c, v = (df[k].values for k in 'ohlcv'); dl = 2 * df.tb.values - v
    cd, cv = np.r_[0, np.cumsum(dl)], np.r_[0, np.cumsum(v)]
    share = lambda a, b: (cd[b] - cd[a]) / max(cv[b] - cv[a], 1e-12)
    res = {(lad, f[0]): [] for lad in LADDERS for f in FILTERS}
    busy = {k: -1 for k in res}
    for i in range(3, len(piv)):
        p1, p2, p3, p4 = piv[i - 3], piv[i - 2], piv[i - 1], piv[i]
        up = p4[2] == 1; s = 1 if up else -1
        if not (s * (p3[1] - p1[1]) > 0 and s * (p4[1] - p2[1]) > 0): continue
        leg = s * (p4[1] - p3[1])
        if leg <= 0: continue
        start = p4[3] + 1
        if start >= len(o): continue
        stop = p3[1] * (1 - s * BUF); tp1, tp2 = p4[1], p4[1] + s * 0.62 * leg
        legd = s * share(p3[0], p4[0] + 1)
        for lad, levels in LADDERS.items():
            lv = [p4[1] - s * f * leg for f in levels]
            if (o[start] <= lv[0]) if up else (o[start] >= lv[0]): continue      # first rung already passed
            inv = p4[1] - s * 0.79 * leg
            d_full = [s * (x - stop) / x for x in lv]
            if min(d_full) <= 0 or max(d_full) > 0.10: continue
            w = 1.0 / len(lv); unit = sum(w * (d + MAKER + TAKER) for d in d_full)
            # walk bars: fills, cancels, exits
            filled = [False] * len(lv); t = start; first_touch = None
            pos = 0.0; cost = 0.0; pnl = 0.0; half = False; cur_stop = stop; done = False; tdone = None
            while t < len(o) and not done:
                if pos == 0 and ((h[t] > p4[1]) if up else (l[t] < p4[1])): break       # new extreme before any fill
                for k, x in enumerate(lv):
                    if not filled[k] and not half and ((l[t] < x * (1 - THRU)) if up else (h[t] > x * (1 + THRU))):
                        if first_touch is None: first_touch = t
                        filled[k] = True; pos += w; cost += w * x
                if pos > 0:
                    avg = cost / pos
                    if (l[t] <= cur_stop) if up else (h[t] >= cur_stop):
                        pnl += pos * (s * (cur_stop - avg) / avg - MAKER - TAKER); pos = 0; done = True; tdone = t; break
                    if t > first_touch and not half and ((h[t] > tp1) if up else (l[t] < tp1)):
                        pnl += pos / 2 * (s * (tp1 - avg) / avg - 2 * MAKER); pos /= 2; cost /= 2; half = True
                        cur_stop = avg * (1 + s * (MAKER + TAKER)); t += 1; continue
                    if t > first_touch and ((h[t] > tp2) if up else (l[t] < tp2)):
                        pnl += pos * (s * (tp2 - avg) / avg - 2 * MAKER); pos = 0; done = True; tdone = t; break
                elif (c[t] < inv) if up else (c[t] > inv):
                    if levels[-1] < 0.79: break                                        # body beyond 0.79: cancel
                t += 1
            if first_touch is None: continue
            if pos > 0:
                tdone = len(o) - 1; avg = cost / pos; pnl += pos * (s * (c[tdone] - avg) / avg - MAKER - TAKER)
            retrd = s * share(p4[0] + 1, first_touch)
            for fname, ta, tc in FILTERS:
                key = (lad, fname)
                if legd <= ta or retrd <= tc or first_touch <= busy[key]: continue
                res[key].append((df.time[first_touch], pnl / unit, sum(filled) / len(lv))); busy[key] = tdone
    return res


if __name__ == '__main__':
    syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
    rows = []
    for sym in syms:
        df = load30(sym)
        for K in (3, 4):
            for (lad, fname), tr in run_sym(df, K).items():
                rows += [(f'K{K}|{lad}|{fname}', sym, t, R, fr) for t, R, fr in tr]
        print(sym, flush=True)
    T = pd.DataFrame(rows, columns=['cfg', 'sym', 'time', 'R', 'fill_frac']); T['test'] = T.time >= SPLIT
    days = (T.time.max() - T.time.min()).days; G = T.groupby('cfg')
    S = pd.DataFrame(dict(n=G.size(), per_day=G.size() / days, win=G.R.apply(lambda x: (x > 0).mean()), avgR=G.R.mean(),
                          first=T[~T.test].groupby('cfg').R.mean(), last=T[T.test].groupby('cfg').R.mean(),
                          coins_pos=T.groupby(['cfg', 'sym']).R.mean().gt(0).groupby('cfg').mean(), fill=G.fill_frac.mean()))
    for f in (0.01, 0.02):
        S = S.join(T.groupby('cfg').apply(lambda x: pd.Series({
            f'x_{f}': np.cumprod(1 + f * x.sort_values('time').R.values)[-1],
            f'mdd_{f}': (lambda e: (e / np.maximum.accumulate(e) - 1).min())(np.cumprod(1 + f * x.sort_values('time').R.values))})))
    S.to_csv('ict_ladder.csv'); pd.set_option('display.width', 250)
    print('positive both halves:', ((S['first'] > 0) & (S['last'] > 0)).sum(), 'of', len(S))
    print(S.sort_values('avgR', ascending=False).head(20).round(3).to_string())
