"""$400 seed, each entry = 10% margin x 10x = 1x seed notional, one add of the same size.
One position at a time across BTC/ETH/SOL, long and short."""
import numpy as np, pandas as pd
SPLIT = pd.Timestamp('2024-04-01', tz='UTC'); SEED = 400; L = 0.10 * 10
A = pd.concat([pd.read_csv(f'retr/avg_{c}.csv', parse_dates=['t_in']).assign(coin=c) for c in ['BTC', 'ETH', 'SOL']])
A = A[A.t_in >= '2021-02-01'].sort_values('t_in')
rules = [c for c in A.columns if c.count('|') == 2]
rows = []
for r in rules:
    outc = pd.to_datetime(A[r + '|out'], utc=True); keep, busy = [], pd.Timestamp('1970', tz='UTC')
    for ti, to, x, ad in zip(A.t_in, outc, A[r], A[r + '|added']):
        if ti >= busy: keep.append((ti, to, x, ad)); busy = to
    K = pd.DataFrame(keep, columns=['t', 'out', 'x', 'added'])
    for per, X in [('train', K[K.t < SPLIT]), ('test', K[K.t >= SPLIT])]:
        eq = np.cumprod(np.maximum(1 + L * X.x.values, 0)); mdd = 1 - (eq / np.maximum.accumulate(eq)).min()
        P = pd.Series(np.r_[1.0, eq], index=pd.DatetimeIndex([X.t.min()] + list(X.out)))
        mr = P.resample('ME').last().ffill().pct_change().dropna()
        days = (X.out.max() - X.t.min()).days
        rows.append(dict(add=r.split('|')[0], stop=r.split('|')[1], target=r.split('|')[2], per=per, trades=len(X),
                         per_day=len(X) / days, win=(X.x > 0).mean(), added=X.added.mean(),
                         avg_win_usd=SEED * L * X.x[X.x > 0].mean(), avg_loss_usd=SEED * L * X.x[X.x <= 0].mean(),
                         month_geo=eq[-1] ** (30 / days) - 1 if eq[-1] > 0 else -1, month_median=mr.median(),
                         month_worst=mr.min(), losing_months=(mr < 0).mean(), maxDD=mdd, final_usd=SEED * eq[-1]))
O = pd.DataFrame(rows); O.to_csv('retr/avgdown_eval.csv', index=False)
W = O.pivot_table(index=['add', 'stop', 'target'], columns='per', values='month_geo').reset_index()
W = W.merge(O[O.per == 'test'], on=['add', 'stop', 'target'])
W['both_pos'] = (W.train > 0) & (W.test > 0)
pd.set_option('display.width', 250)
cols = ['add', 'stop', 'target', 'per_day', 'win', 'added', 'avg_win_usd', 'avg_loss_usd', 'train', 'test',
        'month_median', 'month_worst', 'losing_months', 'maxDD', 'final_usd']
f = lambda X: X[cols].assign(win=(X.win * 100).round(0), added=(X.added * 100).round(0), train=(X.train * 100).round(1),
    test=(X.test * 100).round(1), month_median=(X.month_median * 100).round(1), month_worst=(X.month_worst * 100).round(0),
    losing_months=(X.losing_months * 100).round(0), maxDD=(X.maxDD * 100).round(0), per_day=X.per_day.round(2),
    avg_win_usd=X.avg_win_usd.round(2), avg_loss_usd=X.avg_loss_usd.round(2), final_usd=X.final_usd.round(0))
print('train/test = geometric monthly return %; $ amounts on a $400 seed; test period 2024-04..2026-11')
print(f'\nrules positive in BOTH periods: {W.both_pos.sum()} of {len(W)}  '
      f'(no averaging: {W[W["add"] == "0"].both_pos.sum()} of {(W["add"] == "0").sum()}, '
      f'with averaging: {W[W["add"] != "0"].both_pos.sum()} of {(W["add"] != "0").sum()})')
print('\nTOP 12 by the WORSE of the two periods:')
W['worst_period'] = W[['train', 'test']].min(axis=1)
print(f(W.sort_values('worst_period', ascending=False)).head(12).to_string(index=False))
print('\nmedian monthly return (test) by averaging step:', (W.groupby('add').test.median() * 100).round(2).to_dict())
print('median max drawdown (test) by averaging step:', (W.groupby('add').maxDD.median() * 100).round(0).to_dict())
