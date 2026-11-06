"""Follow-up: best order-flow signals with wider stops (lower fee share) and two fee levels."""
import glob, os, itertools, numpy as np, pandas as pd
from scan_cvd import load, signals
from scan_alts import sim, roll, SPLIT
KEEP = {'absorb|36/0.1', 'absorb|12/0.1', 'diverge|36/0.05', 'diverge|96/0.1', 'imb_momo|12/0.3', 'diverge|96/0.05'}
syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
rows = []
for sym in syms:
    D = load(sym); t, o, h, l, c, v, tb = (D[:, i] for i in range(7)); pc = np.r_[c[0], c[:-1]]
    A = roll(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), 14, 'mean')
    for fam, p, sig in signals(o, h, l, c, v, tb):
        if f'{fam}|{p}' not in KEEP: continue
        for k, b, fee, hold in itertools.product((3.0, 5.0, 8.0), (2.0, 3.0, 5.0), (0.00035, 0.00025), (120, 288)):
            sd = np.nan_to_num(k * A / c); ent, R = sim(sig, sd, b, fee, hold, o, h, l, c)
            rows.append(pd.DataFrame(dict(cfg=f'{fam}|{p}|k{k}|b{b}|f{fee}|h{hold}', sym=sym, t=t[ent], R=R, d=sd[ent - 1])))
    print(sym, flush=True)
T = pd.concat(rows); T['test'] = T.t >= SPLIT; days_out = (T.t.max() - SPLIT) / 864e5
S = T.groupby(['cfg', 'test']).R.agg(n='size', win=lambda x: (x > 0).mean(), avgR='mean').unstack('test')
S.columns = [f'{a}_{"out" if b else "in"}' for a, b in S.columns]
S = S.join(T[T.test].groupby(['cfg', 'sym']).R.mean().gt(0).groupby('cfg').mean().rename('coins_pos_out'))
S = S.join(T[T.test].groupby('cfg').d.median().rename('med_stop')); S['per_day_out'] = S.n_out / days_out
S.to_csv('scan_cvd2.csv'); S['m'] = S[['avgR_in', 'avgR_out']].min(axis=1)
pd.set_option('display.width', 250)
print('configs', len(S), 'positive both', (S.m > 0).sum())
print(S.sort_values('m', ascending=False).head(20).round(4).to_string())
