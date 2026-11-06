"""Multi-coin 5m scalping scan (Binance USDT-M, 2023-11..2026-10).

Goal from owner: 500+ trades/day across coins, win >= 25%, net edge after
market fees. A rule only counts if it works on most coins, not one or two.
Train 2023-11..2025-10, test 2025-11..2026-10 (unseen).
R: loss at stop incl. fees = -1R. Entry next bar open (market), stop before
target inside a bar.
"""
import glob, io, zipfile, itertools, os, numpy as np, pandas as pd
from numba import njit

FEE = 0.00035                       # taker after ~30% rebate
SPLIT = pd.Timestamp('2025-11-01', tz='UTC').value // 10**6
HOLD = 120                          # 10 hours max


def load(sym):
    fr = []
    for f in sorted(glob.glob(f'data/alt5m/{sym}USDT-5m-*.zip')):
        z = zipfile.ZipFile(f); d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]
        fr.append(d.iloc[:, :6].astype(float))
    if not fr: return None
    d = pd.concat(fr); d.columns = list('tohlcv'); d = d[d.t % 300000 == 0].drop_duplicates('t').sort_values('t')
    return d.values


@njit(cache=True)
def sim(sig, stopd, b, fee, hold, o, h, l, c):
    n = len(c); ent = np.empty(n, np.int64); R = np.empty(n); k = 0; i = 0
    while i < n - 2:
        s = sig[i]
        if s == 0 or not (stopd[i] > 0):
            i += 1; continue
        e = i + 1; px = o[e]; d = stopd[i]
        stop = px * (1 - s * d); tgt = px * (1 + s * b * d)
        j_end = min(n - 1, e + hold); out = c[j_end]; j = e
        while j <= j_end:
            if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                out = stop; break
            if (s > 0 and h[j] >= tgt) or (s < 0 and l[j] <= tgt):
                out = tgt; break
            j += 1
        if j > j_end: j = j_end
        R[k] = (s * (out - px) / px - 2 * fee) / (d + 2 * fee); ent[k] = e; k += 1
        i = j + 1
    return ent[:k], R[:k]


def roll(x, n, f):
    return getattr(pd.Series(x).rolling(n, min_periods=n), f)().values


def signals(o, h, l, c, v):
    for n in (12, 36, 96):
        hh = np.r_[np.nan, roll(h, n, 'max')[:-1]]; ll = np.r_[np.nan, roll(l, n, 'min')[:-1]]
        yield 'breakout', n, np.where(c > hh, 1, np.where(c < ll, -1, 0)).astype(np.int8)
        yield 'fade_break', n, np.where(c > hh, -1, np.where(c < ll, 1, 0)).astype(np.int8)
    for n, z in itertools.product((20, 60), (2.0, 3.0)):
        m = roll(c, n, 'mean'); sd = roll(c, n, 'std')
        yield 'band_fade', f'{n}/{z}', np.where(c < m - z * sd, 1, np.where(c > m + z * sd, -1, 0)).astype(np.int8)
        yield 'band_break', f'{n}/{z}', np.where(c > m + z * sd, 1, np.where(c < m - z * sd, -1, 0)).astype(np.int8)
    for f_, s_ in ((9, 50), (20, 200)):
        ef = pd.Series(c).ewm(span=f_).mean().values; es = pd.Series(c).ewm(span=s_).mean().values
        pc = np.r_[c[0], c[:-1]]; pef = np.r_[ef[0], ef[:-1]]
        yield 'pullback', f'{f_}/{s_}', np.where((ef > es) & (pc < pef) & (c > ef), 1,
                                                np.where((ef < es) & (pc > pef) & (c < ef), -1, 0)).astype(np.int8)
    for m_ in (3.0, 5.0):
        av = np.r_[np.nan, roll(v, 60, 'mean')[:-1]]; spike = v > m_ * av; dr = np.sign(c - o)
        yield 'vol_momo', m_, np.where(spike, dr, 0).astype(np.int8)
        yield 'vol_fade', m_, np.where(spike, -dr, 0).astype(np.int8)


if __name__ == '__main__':
    syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
    trades = []                     # (config, sym, entry_time, R, stop)
    for sym in syms:
        D = load(sym)
        if D is None or len(D) < 20000: continue
        t, o, h, l, c, v = (D[:, i] for i in range(6))
        pc = np.r_[c[0], c[:-1]]
        A = roll(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), 14, 'mean')
        for fam, p, sig in signals(o, h, l, c, v):
            for k, b in itertools.product((1.0, 2.0, 3.0), (1.0, 1.5, 2.0, 3.0)):
                sd = np.nan_to_num(k * A / c)
                ent, R = sim(sig, sd, b, FEE, HOLD, o, h, l, c)
                trades.append(pd.DataFrame(dict(cfg=f'{fam}|{p}|k{k}|b{b}', sym=sym, t=t[ent], R=R, d=sd[ent - 1])))
        print(sym, len(D), flush=True)
    T = pd.concat(trades); T['test'] = T.t >= SPLIT
    days_in = (SPLIT - T.t.min()) / 864e5; days_out = (T.t.max() - SPLIT) / 864e5
    g = T.groupby(['cfg', 'test'])
    S = g.R.agg(n='size', win=lambda x: (x > 0).mean(), avgR='mean').unstack('test')
    S.columns = [f'{a}_{"out" if b else "in"}' for a, b in S.columns]
    per_coin = T[T.test].groupby(['cfg', 'sym']).R.mean().gt(0).groupby('cfg').mean().rename('coins_pos_out')
    S = S.join(per_coin).join(T[T.test].groupby('cfg').d.median().rename('med_stop'))
    S['per_day_out'] = S.n_out / days_out
    S.to_csv('scan_alts.csv')
    pd.set_option('display.width', 250)
    print('configs', len(S), '| positive both', ((S.avgR_in > 0) & (S.avgR_out > 0)).sum(),
          '| win>=25% & pos both & >=500/day', ((S.avgR_in > 0) & (S.avgR_out > 0) & (S.win_out >= .25) & (S.per_day_out >= 500)).sum())
    S['m'] = S[['avgR_in', 'avgR_out']].min(axis=1)
    print(S.sort_values('m', ascending=False).head(20).round(4).to_string())
