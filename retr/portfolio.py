"""ONE position at a time across BTC+ETH+SOL, long and short: after a trade closes,
take the next signal from any coin. Position = 40% of seed x 35x = 14x seed (your sizing),
plus Kelly-style smaller sizes for comparison. Reports trades/day and monthly returns."""
import numpy as np, pandas as pd, sys
SIZES = [14.0, 7.0, 4.0, 2.0]
SPLIT = pd.Timestamp('2024-04-01', tz='UTC')
frames = []
for coin in ['BTC', 'ETH', 'SOL']:
    E = pd.read_csv(f'retr/tight_{coin}.csv', parse_dates=['t_in']); E['coin'] = coin; frames.append(E)
A = pd.concat(frames, ignore_index=True)
A = A[A.t_in >= '2021-02-01']                       # all 3 coins trading
rules = [c for c in A.columns if c.count('|') == 1]
rows, monthly = [], {}
for r in rules:
    S = A.sort_values('t_in'); outc = pd.to_datetime(S[r + '|out'], utc=True)
    keep, busy = [], pd.Timestamp('1970', tz='UTC')
    for ti, to, x, cn, d in zip(S.t_in, outc, S[r], S.coin, S.d):
        if ti >= busy: keep.append((ti, to, x, cn, d)); busy = to
    K = pd.DataFrame(keep, columns=['t', 'out', 'x', 'coin', 'd'])
    for per, X in [('train', K[K.t < SPLIT]), ('test', K[K.t >= SPLIT])]:
        days = (X.out.max() - X.t.min()).days
        for L in SIZES:
            eq = np.cumprod(np.maximum(1 + L * X.x.values, 0)); mdd = 1 - (eq / np.maximum.accumulate(eq)).min()
            P = pd.Series(np.r_[1.0, eq], index=pd.DatetimeIndex([X.t.min()] + list(X.out)))
            mo = P.resample('ME').last().ffill(); mr = mo.pct_change().dropna()
            gone = np.argmax(eq < 0.05) if (eq < 0.05).any() else None
            rows.append(dict(stop=r.split('|')[0], target=r.split('|')[1], per=per, L=L, trades=len(X),
                             per_day=len(X) / max(days, 1), win=(X.x > 0).mean(),
                             loss_per_trade_seed=L * X.x[X.x <= 0].mean(), win_per_trade_seed=L * X.x[X.x > 0].mean(),
                             final_x=eq[-1], month_median=mr.median(), month_mean_geo=(eq[-1] ** (30 / max(days, 1)) - 1) if eq[-1] > 0 else -1,
                             month_best=mr.max(), month_worst=mr.min(), losing_months=(mr < 0).mean(), maxDD=mdd,
                             wiped=str(X.out.iloc[gone])[:10] if gone is not None else ''))
            if per == 'test' and L in (14.0, 4.0): monthly[(r, L)] = mr
O = pd.DataFrame(rows); O.to_csv('retr/portfolio.csv', index=False)
pd.set_option('display.width', 250)
fmt = lambda X: X.assign(win=(X.win * 100).round(1), loss_per_trade_seed=(X.loss_per_trade_seed * 100).round(1),
    win_per_trade_seed=(X.win_per_trade_seed * 100).round(1), month_median=(X.month_median * 100).round(1),
    month_mean_geo=(X.month_mean_geo * 100).round(1), month_best=(X.month_best * 100).round(0),
    month_worst=(X.month_worst * 100).round(0), losing_months=(X.losing_months * 100).round(0),
    maxDD=(X.maxDD * 100).round(0), per_day=X.per_day.round(2), final_x=X.final_x.map(lambda v: f'{v:.3g}'))
cols = ['stop', 'target', 'trades', 'per_day', 'win', 'loss_per_trade_seed', 'win_per_trade_seed', 'final_x',
        'month_median', 'month_mean_geo', 'month_worst', 'losing_months', 'maxDD', 'wiped']
for L in SIZES:
    X = O[(O.per == 'test') & (O.L == L)].sort_values('final_x', ascending=False, key=lambda s: s.astype(float))
    print(f'\n===== TEST 2024-04..2026-11 | position {L:.0f}x seed {"(YOUR 40% x 35x)" if L == 14 else ""} | % columns are % of the WHOLE seed')
    print(fmt(X)[cols].head(12).to_string(index=False))
T14 = O[(O.per == 'train') & (O.L == 14.0)]
print('\ntrain-period check (2021-02..2024-03) at 14x for the same rules (final_x):')
print(T14.sort_values('final_x', ascending=False)[['stop', 'target', 'trades', 'win', 'final_x', 'maxDD', 'wiped']].head(12).round(3).to_string(index=False))
best = O[(O.per == 'test') & (O.L == 14.0)].sort_values('final_x', ascending=False).iloc[0]
key = (f"{best.stop}|{best.target}" if best.target == 'pl' else f"{best.stop}|{best.target}", 14.0)
for kk in monthly:
    if kk[0] == f'{best.stop}|{best.target}':
        print(f'\nmonth by month, rule stop {best.stop} target {best.target}, position {kk[1]:.0f}x (TEST):')
        print((monthly[kk] * 100).round(1).to_string())
