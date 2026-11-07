"""Equity for YOUR sizing: 40% of seed as margin x 35x = position 14x seed. One trade at a time per coin."""
import numpy as np, pandas as pd
L = 0.40 * 35
SPLIT = pd.Timestamp('2024-04-01', tz='UTC')
pd.set_option('display.width', 220)
out = []
for coin in ['BTC', 'ETH', 'SOL']:
    E = pd.read_csv(f'retr/tight_{coin}.csv', parse_dates=['time', 't_in'])
    rules = [c for c in E.columns if c.count('|') == 1]
    for d, side in [(1, 'SHORT'), (-1, 'LONG')]:
        S = E[E.d == d].sort_values('t_in')
        for r in rules:
            outc = pd.to_datetime(S[r + '|out'], utc=True)
            keep, busy = [], pd.Timestamp('1970-01-01', tz='UTC')
            for ti, to, x in zip(S.t_in, outc, S[r]):
                if ti >= busy: keep.append((ti, x)); busy = to
            K = pd.DataFrame(keep, columns=['t', 'x'])
            for per, X in [('train', K[K.t < SPLIT]), ('test', K[K.t >= SPLIT])]:
                g = L * X.x                                   # equity change per trade
                eq = np.cumprod(np.maximum(1 + g.values, 0)); mdd = 1 - (eq / np.maximum.accumulate(eq)).min()
                out.append(dict(coin=coin, side=side, stop=r.split('|')[0], target=r.split('|')[1], per=per, n=len(X),
                                win=(X.x > 0).mean(), avg_win_eq=g[g > 0].mean(), avg_loss_eq=g[g <= 0].mean(),
                                E_eq=g.mean(), final_x=eq[-1] if len(eq) else np.nan, maxDD=mdd))
O = pd.DataFrame(out); O.to_csv('retr/tight_eval.csv', index=False)
T = O[O.per == 'test']
print('TEST period 2024-04..2026-11, position = 14x seed. Equity numbers are % of the WHOLE seed per trade.\n')
print(T.sort_values('E_eq', ascending=False).head(12).assign(
    win=lambda x: (x.win * 100).round(1), avg_win_eq=lambda x: (x.avg_win_eq * 100).round(1),
    avg_loss_eq=lambda x: (x.avg_loss_eq * 100).round(2), E_eq=lambda x: (x.E_eq * 100).round(2),
    maxDD=lambda x: (x.maxDD * 100).round(0)).to_string(index=False))
print(f'\nrules with positive expectancy in BOTH train and test: ', end='')
P = O.pivot_table(index=['coin', 'side', 'stop', 'target'], columns='per', values='E_eq')
print(((P.train > 0) & (P.test > 0)).sum(), 'of', len(P))
print('accounts wiped out (equity -> ~0) in test:', (T.final_x < 0.01).sum(), 'of', len(T))
print('\nby stop distance (median over coins/sides/targets, TEST):')
print(T.groupby('stop')[['win', 'avg_loss_eq', 'E_eq', 'maxDD']].median().round(4).to_string())
