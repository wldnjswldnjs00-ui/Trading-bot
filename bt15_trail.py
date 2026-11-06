"""Trailing stop test: ICT 30m ladder (0.5/0.618/0.705) + impulse-delta filter.

Exit modes
  half       — half at prior high, stop to entry, rest to -0.62 ext   (baseline)
  trail      — no target. Stop starts at the prior swing; every time a new swing
               low (long) is confirmed above the current stop, the stop moves up
               just below it. Swings come from a trailing zigzag of size KT.
  half_trail — half at prior high + stop to entry, the rest trails as above.
Stop orders are market (taker). Coins: every file in data/alt5m.
"""
import glob, os, bisect, sys, itertools, numpy as np, pandas as pd
from engine import atr, zigzag
from bt13_ict_cvd import load30, MAKER, TAKER, THRU, BUF, SPLIT

LEVELS = (0.5, 0.618, 0.705)


def run_sym(df, K=3, KTS=(1.5, 2.0, 3.0)):
    A = atr(df); piv = zigzag(df, K, A)
    o, h, l, c, v = (df[k].values for k in 'ohlcv'); dl = 2 * df.tb.values - v
    cd, cv = np.r_[0, np.cumsum(dl)], np.r_[0, np.cumsum(v)]
    share = lambda a, b: (cd[b] - cd[a]) / max(cv[b] - cv[a], 1e-12)
    trails = {kt: zigzag(df, kt, A) for kt in KTS}
    tk = {kt: [q[3] for q in z] for kt, z in trails.items()}
    modes = [('half', None)] + [(m, kt) for m in ('trail', 'half_trail') for kt in KTS]
    res = {m: [] for m in modes}; busy = {m: -1 for m in modes}
    for i in range(3, len(piv)):
        p1, p2, p3, p4 = piv[i - 3], piv[i - 2], piv[i - 1], piv[i]
        up = p4[2] == 1; s = 1 if up else -1
        if not (s * (p3[1] - p1[1]) > 0 and s * (p4[1] - p2[1]) > 0): continue
        leg = s * (p4[1] - p3[1]); start = p4[3] + 1
        if leg <= 0 or start >= len(o): continue
        if s * share(p3[0], p4[0] + 1) <= 0: continue                         # impulse delta filter (A0)
        lv = [p4[1] - s * f * leg for f in LEVELS]
        if (o[start] <= lv[0]) if up else (o[start] >= lv[0]): continue
        stop0 = p3[1] * (1 - s * BUF); tp1, tp2 = p4[1], p4[1] + s * 0.62 * leg
        d_full = [s * (x - stop0) / x for x in lv]
        if min(d_full) <= 0 or max(d_full) > 0.10: continue
        w = 1 / len(lv); unit = sum(w * (d + MAKER + TAKER) for d in d_full)
        inv = p4[1] - s * 0.79 * leg
        for mode, kt in modes:
            if start <= busy[(mode, kt)]: continue
            filled = [False] * 3; pos = cost = pnl = 0.0; half = False; stop = stop0; ft = None; t = start; tend = None
            ti = bisect.bisect_right(tk[kt], start) if kt else 0
            while t < len(o):
                if pos == 0 and (((h[t] > p4[1]) if up else (l[t] < p4[1])) or ((c[t] < inv) if up else (c[t] > inv))): break
                for k, x in enumerate(lv):
                    if not filled[k] and not half and ((l[t] < x * (1 - THRU)) if up else (h[t] > x * (1 + THRU))):
                        filled[k] = True; pos += w; cost += w * x; ft = ft if ft is not None else t
                if pos > 0:
                    avg = cost / pos
                    trailing = mode == 'trail' or (mode == 'half_trail' and half)
                    if trailing:
                        while ti < len(trails[kt]) and trails[kt][ti][3] <= t:
                            q = trails[kt][ti]; ti += 1
                            if q[2] == (-1 if up else 1) and q[0] >= ft and s * (q[1] * (1 - s * BUF) - stop) > 0:
                                stop = q[1] * (1 - s * BUF)
                    if (l[t] <= stop) if up else (h[t] >= stop):
                        pnl += pos * (s * (stop - avg) / avg - MAKER - TAKER); pos = 0; tend = t; break
                    if mode in ('half', 'half_trail') and t > ft and not half and ((h[t] > tp1) if up else (l[t] < tp1)):
                        pnl += pos / 2 * (s * (tp1 - avg) / avg - 2 * MAKER); pos /= 2; cost /= 2; half = True
                        stop = max(stop, avg * (1 + MAKER + TAKER)) if up else min(stop, avg * (1 - MAKER - TAKER))
                        t += 1; continue
                    if mode == 'half' and t > ft and ((h[t] > tp2) if up else (l[t] < tp2)):
                        pnl += pos * (s * (tp2 - avg) / avg - 2 * MAKER); pos = 0; tend = t; break
                t += 1
            if ft is None: continue
            if pos > 0: tend = len(o) - 1; avg = cost / pos; pnl += pos * (s * (c[tend] - avg) / avg - MAKER - TAKER)
            res[(mode, kt)].append((df.time[ft], pnl / unit, tend - ft)); busy[(mode, kt)] = tend
    return res


if __name__ == '__main__':
    syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
    if len(sys.argv) > 1: syms = [s for s in syms if s in set(open(sys.argv[1]).read().split())]
    rows = []
    for sym in syms:
        try: df = load30(sym)
        except Exception: continue
        if len(df) < 2000: continue
        for (mode, kt), tr in run_sym(df).items():
            rows += [(f'{mode}|{kt}', sym, t, R, bars) for t, R, bars in tr]
        print(sym, flush=True)
    T = pd.DataFrame(rows, columns=['cfg', 'sym', 'time', 'R', 'bars']); T['test'] = T.time >= SPLIT
    days = (T.time.max() - T.time.min()).days; G = T.groupby('cfg')
    win = T[T.R > 0].groupby('cfg').R.mean(); loss = -T[T.R <= 0].groupby('cfg').R.mean()
    S = pd.DataFrame(dict(n=G.size(), per_day=G.size() / days, win=G.R.apply(lambda x: (x > 0).mean()), avg_win=win, avg_loss=loss,
                          payoff=win / loss, avgR=G.R.mean(), first=T[~T.test].groupby('cfg').R.mean(), last=T[T.test].groupby('cfg').R.mean(),
                          coins_pos=T.groupby(['cfg', 'sym']).R.mean().gt(0).groupby('cfg').mean(), hold_h=G.bars.median() / 2,
                          top1pct_share=G.R.apply(lambda x: x.nlargest(max(1, len(x) // 100)).sum() / x[x > 0].sum())))
    for f in (0.01, 0.02):
        S = S.join(T.groupby('cfg').apply(lambda x: pd.Series({
            f'x_{f}': np.cumprod(1 + f * x.sort_values('time').R.values)[-1],
            f'mdd_{f}': (lambda e: (e / np.maximum.accumulate(e) - 1).min())(np.cumprod(1 + f * x.sort_values('time').R.values))})))
    out = 'trail_%d.csv' % len(syms); S.to_csv(out); pd.set_option('display.width', 260)
    print('coins', len(syms)); print(S.round(3).to_string())
