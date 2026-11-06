"""Report for full_map: overall, by setup kind / pattern, BTC vs coins, and the
regime-button overlay (perfect caller vs 7 / 14 days late)."""
import numpy as np, pandas as pd

T = pd.read_csv('ew/full_map_trades.csv.gz'); T['time'] = pd.to_datetime(T.time, utc=True)
T['univ'] = np.where(T.sym == 'BTC9y', 'BTC 9y', 'coins 3y')
B = np.load('data/btc_spot_1m.npy'); b = pd.Series(B[:, 4], index=pd.to_datetime(B[:, 0].astype(np.int64), unit='ms', utc=True))
mret = b.resample('ME').last().pct_change()


def regime_at(times, lag_days):
    """+1/-1/0: BTC calendar-month move > +-8% — the month the caller says, applied `lag_days` late."""
    t = times - pd.Timedelta(days=lag_days)
    m = t.dt.tz_convert(None).dt.to_period('M').dt.to_timestamp('M')
    r = mret.copy(); r.index = r.index.tz_convert(None).normalize()
    v = m.dt.normalize().map(r).fillna(0).values
    return np.where(v > 0.08, 1, np.where(v < -0.08, -1, 0))


pd.set_option('display.width', 240); out = []
for u, X in T.groupby('univ'):
    days = (X.time.max() - X.time.min()).days
    g = X.groupby(['stop', 'tf']).agg(n=('R', 'size'), win=('R', lambda x: (x > 0).mean()), avgR=('R', 'mean'))
    g['per_day'] = g.n / days; g['R_month'] = g.avgR * g.per_day * 30.4
    out.append(f'=== {u} ===\n' + g.round(3).to_string())
    out.append('\nby setup kind\n' + X.groupby(['stop', 'kind']).R.agg(['size', 'mean']).unstack('stop').round(3).to_string())
    k = X[X.stop == 'invalid'].groupby(['pattern', 'wave', 'kind']).R.agg(['size', 'mean'])
    out.append('\nby pattern (stop = invalid, n>=50)\n' + k[k['size'] >= 50].sort_values('mean', ascending=False).round(3).to_string())
    if u == 'coins 3y':
        cp = X[X.stop == 'invalid'].groupby(['tf', 'sym']).R.mean().gt(0).groupby('tf').mean()
        out.append('\nshare of coins with positive avgR (stop = invalid)\n' + cp.round(3).to_string())
for lag in (0, 7, 14):
    T[f'reg{lag}'] = regime_at(T.time, lag)
rows = []
for (u, tf, stop), X in T.groupby(['univ', 'tf', 'stop']):
    days = max((X.time.max() - X.time.min()).days, 1)
    r = dict(univ=u, tf=tf, stop=stop, all=X.R.mean())
    for lag in (0, 7, 14):
        W = X[X[f'reg{lag}'] == X.side]
        r[f'with_lag{lag}'] = W.R.mean(); r[f'n_day_lag{lag}'] = len(W) / days
        r[f'Rmonth_lag{lag}'] = W.R.mean() * len(W) / days * 30.4
    rows.append(r)
out.append('\nREGIME BUTTON OVERLAY: keep only trades on the side of the month (lag 0 = perfect caller)\n' +
           pd.DataFrame(rows).round(3).to_string(index=False))
txt = '\n'.join(out); print(txt); open('logs/ew_full_map_report.txt', 'w').write(txt)
