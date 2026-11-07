"""Score every (detection setting x side x trade rule) of ob.py.
train < 2024-04-01 <= test (same date for every coin). A rule SURVIVES if mean R > 0
in train, in test, and in both halves of test, with >= 50 test trades. Survivors are
then checked on the OTHER coins (never used to pick them) and sized with Kelly."""
import numpy as np, pandas as pd, glob, sys
coins = sys.argv[1].split(',') if len(sys.argv) > 1 else ['BTC']
E = pd.concat([pd.read_csv(f'retr/ob_events_{c}.csv', parse_dates=['time']) for c in coins], ignore_index=True)
RC = [c for c in E.columns if c.count('|') == 3]
SPLIT, MID = pd.Timestamp('2024-04-01', tz='UTC'), pd.Timestamp('2025-07-15', tz='UTC')
E['per'] = np.select([E.time < SPLIT, E.time < MID], ['tr', 'te1'], 'te2')
rows = []
for (K, VM, P, VS, d), G in E.groupby(['K', 'VM', 'P', 'VS', 'd']):
    for col in RC:
        x = G[['per', col]].dropna()
        tr, t1, t2 = (x[x.per == p][col] for p in ('tr', 'te1', 'te2'))
        te = pd.concat([t1, t2])
        if len(tr) < 50 or len(te) < 30: continue
        rows.append(dict(side='SHORT' if d == 1 else 'LONG', K=K, VM=VM, P=P, VS=VS, rule=col,
                         n_tr=len(tr), R_tr=tr.mean(), n_te=len(te), R_te=te.mean(), R_te1=t1.mean(), R_te2=t2.mean(),
                         win_te=(te > 0).mean(), avgwin=te[te > 0].mean(), avgloss=te[te <= 0].mean()))
O = pd.DataFrame(rows)
O.to_csv(f"retr/ob_scores_{'_'.join(coins)}.csv", index=False)
pd.set_option('display.width', 260)
print(f'coins {coins}: {len(E):,} setups, {len(O):,} rules scored')
for sd in ['SHORT', 'LONG']:
    X = O[O.side == sd]
    surv = X[(X.R_tr > 0) & (X.R_te > 0) & (X.R_te1 > 0) & (X.R_te2 > 0) & (X.n_te >= 50)]
    luck = ((X.R_tr > 0) & (X.n_te >= 50)).sum() / 8
    print(f'\n===== {sd}: {len(X)} rules | survivors {len(surv)} | pure luck would give ~{luck:.0f}')
    print('  does VOLUME matter? median test R by top-candle volume filter VM:', X.groupby('VM').R_te.median().round(3).to_dict())
    print('  by impulse-volume filter VS:', X.groupby('VS').R_te.median().round(3).to_dict())
    print('  by entry:', X.assign(e=X.rule.str.split('|').str[0]).groupby('e').R_te.median().round(3).to_dict(),
          '| by stop:', X.assign(e=X.rule.str.split('|').str[1]).groupby('e').R_te.median().round(3).to_dict(),
          '| by target:', X.assign(e=X.rule.str.split('|').str[2]).groupby('e').R_te.median().round(3).to_dict())
    print('  best 8 by TRAIN -> TEST:')
    print(X.sort_values('R_tr', ascending=False).head(8).round(3).to_string(index=False))
    if len(surv):
        print('  survivors (best 12 by test R):')
        print(surv.sort_values('R_te', ascending=False).head(12).round(3).to_string(index=False))
