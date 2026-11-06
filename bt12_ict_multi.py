"""ICT (best BTC settings) on 30 coins, futures 5m data 2023-11..2026-10.
Market (bt8_ict) and conservative limit (bt10_limit) entries.
Select nothing new: settings were chosen on BTC 2017-2023, so every coin and
the whole period here are unseen. Split still shown: 2023-11..2025-10 / 2025-11..
"""
import glob, os, zipfile, numpy as np, pandas as pd
import bt8_ict as I, bt10_limit as L

CFGS = [('30min', None, 4, 0.705, 'half'), ('30min', None, 6, 0.705, 'half'), ('30min', None, 3, 0.5, 'half'),
        ('30min', '12h', 6, 0.705, 'full'), ('15min', '4h', 6, 0.705, 'half')]
SPLIT = pd.Timestamp('2025-11-01', tz='UTC')


def load(sym):
    fr = []
    for f in sorted(glob.glob(f'data/alt5m/{sym}USDT-5m-*.zip')):
        z = zipfile.ZipFile(f); d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]
        fr.append(d.iloc[:, :6].astype(float))
    d = pd.concat(fr); d.columns = ['t', 'o', 'h', 'l', 'c', 'v']; d = d[d.t % 300000 == 0].drop_duplicates('t')
    d['time'] = pd.to_datetime(d.t, unit='ms', utc=True)
    return d.set_index('time')[['o', 'h', 'l', 'c', 'v']].sort_index()


syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
allT = []
for sym in syms:
    I.B5 = load(sym)
    for ltf, htf, K, level, mode in CFGS:
        df = I.bars(ltf); bias = I.htf_bias(htf, 3, df.time) if htf else None
        name = f'{ltf}|{htf or "-"}|K{K}|{level}|{mode}'
        T = I.run(df, I.setups(df, K, level), bias, mode, 0.00035); T['entry'] = 'market'
        S, placed = L.orders(df, K, level)
        U = L.run(df, S, bias, mode); U['entry'] = 'limit'
        for X in (T, U):
            X['cfg'] = name; X['sym'] = sym; allT.append(X)
    print(sym, flush=True)
A = pd.concat(allT); A['test'] = A.time >= SPLIT
days = (A.time.max() - A.time.min()).days
G = A.groupby(['cfg', 'entry'])
S = pd.DataFrame(dict(n=G.size(), per_day=G.size() / days, win=G.R.apply(lambda x: (x > 0).mean()), avgR=G.R.mean(),
                      avgR_first=A[~A.test].groupby(['cfg', 'entry']).R.mean(), avgR_last=A[A.test].groupby(['cfg', 'entry']).R.mean(),
                      coins_pos=A.groupby(['cfg', 'entry', 'sym']).R.mean().gt(0).groupby(['cfg', 'entry']).mean()))
for f in (0.005, 0.01):
    def grow(x):
        x = x.sort_values('time'); g = 1 + f * x.R.values; eq = np.cumprod(g)
        mon = pd.Series(g, index=x.time.values).groupby(x.time.dt.strftime('%Y-%m').values).prod() - 1
        return pd.Series({f'x_{f}': eq[-1], f'mdd_{f}': (eq / np.maximum.accumulate(eq) - 1).min(), f'med_month_{f}': mon.median()})
    S = S.join(A.groupby(['cfg', 'entry']).apply(grow))
S.to_csv('ict_multi.csv'); A.to_csv('ict_multi_trades.csv.gz', index=False)
pd.set_option('display.width', 250); print(S.round(3).to_string())
