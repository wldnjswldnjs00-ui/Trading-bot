"""Kelly-optimal position size (x seed) for the tight-stop rules: chosen on TRAIN,
applied on TEST. Only rules positive in train on all 3 coins for that side."""
import numpy as np, pandas as pd
SPLIT = pd.Timestamp('2024-04-01', tz='UTC')
O = pd.read_csv('retr/tight_eval.csv')
def trades(coin, side, rule):
    E = pd.read_csv(f'retr/tight_{coin}.csv', parse_dates=['t_in'])
    S = E[E.d == (1 if side == 'SHORT' else -1)].sort_values('t_in')
    outc = pd.to_datetime(S[rule + '|out'], utc=True)
    keep, busy = [], pd.Timestamp('1970-01-01', tz='UTC')
    for ti, to, x in zip(S.t_in, outc, S[rule]):
        if ti >= busy: keep.append((ti, x)); busy = to
    return pd.DataFrame(keep, columns=['t', 'x'])
Ls = np.arange(0.5, 40.01, 0.5)
P = O[O.per == 'train'].groupby(['side', 'stop', 'target']).E_eq.agg(['min', 'count']).reset_index()
P = P[(P['count'] == 3) & (P['min'] > 0)]
print('rules positive in TRAIN on BTC, ETH and SOL:', len(P))
rows = []
for _, r in P.iterrows():
    rule = f"{r['stop']}|{r['target']}"
    for coin in ['BTC', 'ETH', 'SOL']:
        K = trades(coin, r.side, rule)
        tr, te = K[K.t < SPLIT].x.values, K[K.t >= SPLIT].x.values
        g = [np.mean(np.log(np.maximum(1 + L * tr, 1e-9))) for L in Ls]
        Lk = Ls[int(np.argmax(g))]
        for nm, Lx in [('full', Lk), ('half', Lk / 2), ('yours', 14.0)]:
            eq = np.cumprod(np.maximum(1 + Lx * te, 0)); mdd = 1 - (eq / np.maximum.accumulate(eq)).min()
            yrs = (K.t.max() - SPLIT).days / 365
            rows.append(dict(side=r.side, rule=rule, coin=coin, sizing=nm, L=Lx, test_x=eq[-1],
                             CAGR=eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1, maxDD=mdd,
                             per_month=(eq[-1] ** (1 / (yrs * 12)) - 1) if eq[-1] > 0 else -1))
R = pd.DataFrame(rows); R.to_csv('retr/tight_kelly.csv', index=False)
pd.set_option('display.width', 200)
for sz in ['full', 'half', 'yours']:
    X = R[R.sizing == sz]
    print(f'\n== sizing: {sz}  (median over rules x coins)  position x seed {X.L.median():.1f} | test 2.6y: x{X.test_x.median():.2f} '
          f'| per month {X.per_month.median():+.1%} | max DD {X.maxDD.median():.0%} | wiped out {(X.test_x < 0.05).mean():.0%}')
best = R[R.sizing == 'half'].groupby(['side', 'rule']).agg(L=('L', 'median'), test_x_min=('test_x', 'min'),
       test_x_med=('test_x', 'median'), mdd=('maxDD', 'max'), pm=('per_month', 'median')).sort_values('test_x_min', ascending=False)
print('\nHALF-Kelly, rules ranked by their WORST coin in test:')
print(best.head(10).round(3).to_string())
