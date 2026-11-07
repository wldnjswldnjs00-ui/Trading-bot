import numpy as np, pandas as pd
import glob
E = pd.concat([pd.read_csv(f, parse_dates=['time']) for f in sorted(glob.glob('retr/zone2_events_*.csv'))], ignore_index=True).sort_values('time').reset_index(drop=True)
print('coins:', E.coin.value_counts().to_dict())
VC = [c for c in E.columns if c.count('|') == 3]
split = E.time.iloc[int(len(E) * 0.6)]
E['half'] = np.where(E.time < split, 'train', 'test')
mid = E[E.half == 'test'].time.median()
FILT = {'all': np.ones(len(E), bool)}
for k in ['imp_atr', 'bounce', 'vr']:
    q = E[E.half == 'train'][k].quantile([1 / 3, 2 / 3]).values
    FILT[f'{k}<={q[0]:.2f}'] = (E[k] <= q[0]).values
    FILT[f'{k}>{q[1]:.2f}'] = (E[k] > q[1]).values
FILT['with_trend'] = (E.trend == 1).values; FILT['against_trend'] = (E.trend == 0).values
rows = []
for d, sd in [(-1, 'SHORT'), (1, 'LONG')]:
    for fn, fm in FILT.items():
        X = E[(E.d == d).values & fm]
        for col in VC:
            tr = X[X.half == 'train'][col].dropna(); te = X[X.half == 'test'][col].dropna()
            if len(tr) < 30 or len(te) < 20: continue
            T = X[X.half == 'test']
            rows.append(dict(side=sd, filter=fn, variant=col, n_tr=len(tr), R_tr=tr.mean(), n_te=len(te), R_te=te.mean(),
                             R_te1=T[T.time < mid][col].dropna().mean(), R_te2=T[T.time >= mid][col].dropna().mean(),
                             win_te=(te > 0).mean(), avgwin_te=te[te > 0].mean(), avgloss_te=te[te <= 0].mean()))
O = pd.DataFrame(rows); O.to_csv('retr/zone2_scores.csv', index=False)
pd.set_option('display.width', 250)
print(f'{len(O)} scored combos (side x filter x variant) with enough trades')
for sd in ['SHORT', 'LONG']:
    X = O[O.side == sd]
    surv = X[(X.R_tr > 0) & (X.R_te > 0) & (X.R_te1 > 0) & (X.R_te2 > 0) & (X.n_te >= 30)]
    luck = ((X.R_tr > 0) & (X.n_te >= 30)).sum() / 8
    print(f'\n===== {sd}: {len(X)} combos | survivors {len(surv)} | luck alone would give ~{luck:.0f}')
    print('best 10 on TRAIN -> TEST:')
    print(X.sort_values('R_tr', ascending=False).head(10).round(2).to_string(index=False))
    if len(surv):
        print('survivors:'); print(surv.sort_values('R_te', ascending=False).head(15).round(2).to_string(index=False))
    print('\nby entry type (median test R over all combos):', X.assign(e=X.variant.str.split('|').str[1]).groupby('e').R_te.median().round(3).to_dict())
    print('by zone      :', X.assign(e=X.variant.str.split('|').str[0]).groupby('e').R_te.median().round(3).to_dict())
    print('by breakeven :', X.assign(e=X.variant.str.split('|').str[3]).groupby('e').R_te.median().round(3).to_dict())
