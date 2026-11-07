"""Kelly sizing + equity simulation for one rule, realistic:
 - one position at a time per coin (skip setups while a trade is open)
 - skip trades whose stop distance < MINRISK (fees would dominate)
 - position = equity * f / risk, capped at LEVCAP x equity (then risk/trade is smaller)
usage: python retr/ob_kelly.py <trades csv> [<trades csv> ...]"""
import numpy as np, pandas as pd, sys
RULE, MINRISK, LEVCAP = 'near|sh|pl|be0', 0.0015, 20
for f_ in sys.argv[1:]:
    E = pd.read_csv(f_, parse_dates=['time', 't_in', 't_out']).dropna(subset=[RULE, 't_in'])
    E = E[E.risk >= MINRISK].sort_values('t_in')
    keep, busy = [], pd.Timestamp('1970-01-01', tz='UTC')
    for _, r in E.iterrows():
        if r.t_in >= busy: keep.append(r); busy = r.t_out
    T = pd.DataFrame(keep)
    SPLIT = pd.Timestamp('2024-04-01', tz='UTC')
    print(f'\n### {f_}  trades kept {len(T)} ({T.t_in.min():%Y-%m}..{T.t_in.max():%Y-%m})')
    for nm, X in [('train', T[T.t_in < SPLIT]), ('test', T[T.t_in >= SPLIT]), ('all', T)]:
        R = X[RULE]; p = (R > 0).mean(); W = R[R > 0].mean(); Lq = -R[R <= 0].mean()
        b = W / Lq; kelly = p - (1 - p) / b
        print(f'  {nm:<5} n={len(X):>4} win {p:.1%} avgwin {W:+.2f}R avgloss {-Lq:+.2f}R  E={R.mean():+.3f}R  Kelly f*={kelly:+.1%}  trades/yr {len(X) / ((X.t_in.max() - X.t_in.min()).days / 365):.0f}')
    Te = T[T.t_in >= SPLIT]
    print('  TEST-period equity, compounding (risk per trade f):')
    for f in [0.0025, 0.005, 0.01, 0.02, 0.03]:
        eq, peak, mdd, path = 1.0, 1.0, 0.0, []
        for _, r in Te.iterrows():
            lev = min(f / r.risk, LEVCAP)
            eq *= max(1 + lev * r.risk * r[RULE], 0.0)
            peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak); path.append((r.t_out, eq))
        P = pd.Series([x[1] for x in path], index=pd.DatetimeIndex([x[0] for x in path]))
        mo = P.resample('ME').last().ffill(); mr = mo.pct_change().dropna()
        yrs = (Te.t_out.max() - Te.t_in.min()).days / 365
        print(f'    f={f:>5.2%}: x{eq:8.2f} over {yrs:.1f}y | CAGR {eq ** (1 / yrs) - 1:+7.1%} | median month {mr.median():+6.1%} '
              f'| worst month {mr.min():+6.1%} | max DD {mdd:5.1%} | losing months {(mr < 0).mean():.0%}')
