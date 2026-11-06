"""Order-flow (CVD) scalping scan on 30 coins, 5m, 2023-11..2026-10.

Binance klines carry taker-buy volume, so per bar:
  delta = taker buy - taker sell = 2 * taker_buy - volume   (CVD = running sum)
Families:
  imb_momo / imb_fade — delta share of volume over n bars beyond +-thr: follow / fade
  absorb              — price at a new n-bar low but buyers dominate the bar (and mirror)
  diverge             — price down over n bars while net delta over n bars is up (and mirror)
Same simulator, fees, split and R definition as scan_alts.py.
"""
import glob, io, zipfile, itertools, os, numpy as np, pandas as pd
from scan_alts import sim, roll, FEE, SPLIT, HOLD


def load(sym):
    fr = []
    for f in sorted(glob.glob(f'data/alt5m/{sym}USDT-5m-*.zip')):
        z = zipfile.ZipFile(f); d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]
        fr.append(d.iloc[:, [0, 1, 2, 3, 4, 5, 9]].astype(float))
    if not fr: return None
    d = pd.concat(fr); d.columns = ['t', 'o', 'h', 'l', 'c', 'v', 'tb']
    d = d[d.t % 300000 == 0].drop_duplicates('t').sort_values('t')
    return d.values


def signals(o, h, l, c, v, tb):
    delta = 2 * tb - v
    for n, thr in itertools.product((1, 3, 12), (0.3, 0.5)):
        r = roll(delta, n, 'sum') / np.maximum(roll(v, n, 'sum'), 1e-12)
        s = np.where(r > thr, 1, np.where(r < -thr, -1, 0)).astype(np.int8)
        yield 'imb_momo', f'{n}/{thr}', s
        yield 'imb_fade', f'{n}/{thr}', (-s).astype(np.int8)
    r1 = delta / np.maximum(v, 1e-12)
    for n, thr in itertools.product((12, 36), (0.1, 0.2)):
        hh = np.r_[np.nan, roll(h, n, 'max')[:-1]]; ll = np.r_[np.nan, roll(l, n, 'min')[:-1]]
        yield 'absorb', f'{n}/{thr}', np.where((l < ll) & (r1 > thr), 1, np.where((h > hh) & (r1 < -thr), -1, 0)).astype(np.int8)
    for n, thr in itertools.product((12, 36, 96), (0.05, 0.1)):
        pchg = c - np.r_[np.full(n, np.nan), c[:-n]]
        dn = roll(delta, n, 'sum') / np.maximum(roll(v, n, 'sum'), 1e-12)
        yield 'diverge', f'{n}/{thr}', np.where((pchg < 0) & (dn > thr), 1, np.where((pchg > 0) & (dn < -thr), -1, 0)).astype(np.int8)


if __name__ == '__main__':
    syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
    trades = []
    for sym in syms:
        D = load(sym)
        if D is None or len(D) < 20000: continue
        t, o, h, l, c, v, tb = (D[:, i] for i in range(7))
        pc = np.r_[c[0], c[:-1]]
        A = roll(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), 14, 'mean')
        for fam, p, sig in signals(o, h, l, c, v, tb):
            for k, b in itertools.product((1.0, 2.0, 3.0), (1.0, 1.5, 2.0, 3.0)):
                sd = np.nan_to_num(k * A / c)
                ent, R = sim(sig, sd, b, FEE, HOLD, o, h, l, c)
                trades.append(pd.DataFrame(dict(cfg=f'{fam}|{p}|k{k}|b{b}', sym=sym, t=t[ent], R=R, d=sd[ent - 1],
                                                gross=R * (sd[ent - 1] + 2 * FEE) / np.maximum(sd[ent - 1], 1e-12) + 2 * FEE / np.maximum(sd[ent - 1], 1e-12))))
        print(sym, flush=True)
    T = pd.concat(trades); T['test'] = T.t >= SPLIT
    days_out = (T.t.max() - SPLIT) / 864e5
    S = T.groupby(['cfg', 'test']).agg(n=('R', 'size'), win=('R', lambda x: (x > 0).mean()), avgR=('R', 'mean'),
                                       grossR=('gross', 'mean')).unstack('test')
    S.columns = [f'{a}_{"out" if b else "in"}' for a, b in S.columns]
    S = S.join(T[T.test].groupby(['cfg', 'sym']).R.mean().gt(0).groupby('cfg').mean().rename('coins_pos_out'))
    S['per_day_out'] = S.n_out / days_out
    S.to_csv('scan_cvd.csv')
    pd.set_option('display.width', 250)
    print('configs', len(S), '| positive both', ((S.avgR_in > 0) & (S.avgR_out > 0)).sum(),
          '| gross positive both', ((S.grossR_in > 0) & (S.grossR_out > 0)).sum())
    S['m'] = S[['avgR_in', 'avgR_out']].min(axis=1)
    print(S.sort_values('m', ascending=False).head(20).round(4).to_string())
