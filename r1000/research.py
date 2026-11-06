"""1000%/month research — honest walk-forward on BTCUSDT 1h (2017-08 .. 2026-10).

Rules that stop the backtest from lying:
  * signals use only closed bars; position at bar t is decided at close of t-1
  * params chosen on TRAIN (2017-2022), judged once on TEST (2023-2026-10)
  * costs: OKX taker 0.05% + slippage 0.02% per side per unit turnover,
    funding 0.01% per 8h on gross exposure
  * leverage sims liquidate the account when an intrabar move wipes the margin
"""
import numpy as np, pandas as pd, itertools, sys

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'data/btcusdt_1h.csv')
df['time'] = pd.to_datetime(df.time)
df = df.set_index('time')
c, h, l = df.c, df.h, df.l
r = c.pct_change().fillna(0)
COST = 0.0007
FUND = 0.0001 / 8
SPLIT = '2023-01-01'
tr = df.index < SPLIT

def state(entry_long, entry_short, exit_long=None, exit_short=None):
    """+1/-1/0 state machine from boolean events (vectorised via ffill)."""
    s = pd.Series(np.nan, index=df.index)
    if exit_long is not None:
        s[exit_long] = 0; s[exit_short] = 0
    s[entry_long] = 1; s[entry_short] = -1
    return s.ffill().fillna(0)

def pnl(pos):
    pos = pos.shift(1).fillna(0)                      # decided at previous close
    return pos * r - COST * pos.diff().abs().fillna(0) - FUND * pos.abs()

def stats(x):
    if x.std() == 0: return dict(sharpe=0, ann=0, hit=0)
    return dict(sharpe=x.mean() / x.std() * np.sqrt(24 * 365),
                ann=x.mean() * 24 * 365, n=len(x))

S = {}
for n in [24, 48, 96, 168, 336, 720]:
    hh, ll = h.rolling(n).max().shift(1), l.rolling(n).min().shift(1)
    xh, xl = h.rolling(n // 2).max().shift(1), l.rolling(n // 2).min().shift(1)
    S[f'donchian_{n}'] = state(c > hh, c < ll, c < xl, c > xh)
for f, s_ in [(12, 48), (24, 96), (48, 192), (96, 384), (168, 720)]:
    S[f'ema_{f}_{s_}'] = np.sign(c.ewm(span=f).mean() - c.ewm(span=s_).mean())
for n in [24, 72, 168, 336, 720]:
    S[f'tsmom_{n}'] = np.sign(c.pct_change(n))
for n, k in itertools.product([24, 48, 168], [1.5, 2.0, 2.5, 3.0]):
    m, sd = c.rolling(n).mean(), c.rolling(n).std()
    z = (c - m) / sd
    S[f'meanrev_{n}_{k}'] = state(z < -k, z > k, z > 0, z < 0)
# hour-of-week seasonality, learned on TRAIN only
how = df.index.dayofweek * 24 + df.index.hour
mu = r[tr].groupby(how[tr]).mean().shift(0)
nxt = pd.Series(how, index=df.index).map(lambda x: x)  # position for bar t uses bar t's hour
for q in [0.0, 0.0002, 0.0004]:
    sig = pd.Series(how, index=df.index).map(lambda x: np.sign(mu.get(x, 0)) if abs(mu.get(x, 0)) > q else 0)
    S[f'season_{q}'] = sig.shift(-1).fillna(0)          # pnl() shifts back: trade bar t with hour(t)

rows = []
for k, pos in S.items():
    p = pnl(pos)
    a, b = stats(p[tr]), stats(p[~tr])
    turn = pos.diff().abs().sum() / (len(pos) / 24 / 365)
    rows.append(dict(strat=k, train_sharpe=a['sharpe'], test_sharpe=b['sharpe'],
                     test_ann_1x=b['ann'], trades_per_yr=turn / 2))
R = pd.DataFrame(rows).sort_values('train_sharpe', ascending=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))

# --- choose on TRAIN, judge on TEST: top-5 train Sharpe, equal-weight ensemble
top = R.strat.head(5).tolist()
ens = sum(S[k] for k in top) / len(top)
pe = pnl(ens)
print('\nENSEMBLE of top-5 by TRAIN sharpe:', top)
print(' train', {k: round(v, 3) for k, v in stats(pe[tr]).items()})
print(' test ', {k: round(v, 3) for k, v in stats(pe[~tr]).items()})
pe.to_frame('ret').to_csv('r1000/ensemble_returns.csv')

# --- leverage sweep on TEST with liquidation (bar low/high vs margin)
pos = ens.shift(1).fillna(0)[~tr]
rt = r[~tr]
adverse = np.where(pos > 0, (l / c.shift(1) - 1)[~tr], (h / c.shift(1) - 1)[~tr] * -1)
adverse = np.where(pos == 0, 0, adverse * np.abs(pos))
bar = pe[~tr]
months = bar.index.to_period('M')
print('\nLEVERAGE on TEST (2023-01..2026-10), monthly reset NOT used (true compounding):')
print(f"{'lev':>4} {'final_x':>12} {'median_mo':>10} {'best_mo':>9} {'worst_mo':>9} {'mo>=1000%':>9} {'liquidated':>10}")
for L in [1, 2, 3, 5, 10, 20, 50, 100]:
    eq, liq, path = 1.0, None, []
    for t, (ret, adv) in enumerate(zip(bar.values, adverse)):
        if L * adv <= -0.99:                         # margin gone inside the bar
            liq = bar.index[t]; eq = 0; path.append(eq); break
        eq *= max(1 + L * ret, 0); path.append(eq)
        if eq <= 0.01: liq = bar.index[t]; eq = 0; break
    ps = pd.Series(path, index=bar.index[:len(path)])
    mo = ps.groupby(months[:len(ps)]).last()
    mr = mo.pct_change().dropna()
    print(f"{L:>4} {eq:>12.3g} {mr.median():>10.1%} {mr.max():>9.1%} {mr.min():>9.1%} {(mr >= 10).mean():>9.0%} {str(liq)[:10] if liq else '-':>10}")

# --- what hourly direction accuracy would 1000%/month need? (noisy oracle)
rng = np.random.default_rng(0)
rr = rt.values
print('\nNOISY ORACLE on TEST: robot calls each 1h candle direction with accuracy p, 1x lev, after costs')
for p in [0.50, 0.51, 0.52, 0.53, 0.55, 0.60]:
    out = []
    for _ in range(20):
        ok = rng.random(len(rr)) < p
        sgn = np.where(ok, np.sign(rr), -np.sign(rr))
        x = sgn * rr - COST * np.abs(np.diff(np.r_[0, sgn])) - FUND
        x = pd.Series(x, index=rt.index)
        m = x.groupby(months).sum()
        mu_, var_ = x.mean(), x.var()
        kelly = mu_ / var_ if var_ > 0 else 0
        g_month = (mu_ * kelly - 0.5 * kelly**2 * var_) * 24 * 30 if kelly > 0 else 0
        out.append((m.mean(), kelly, np.expm1(g_month)))
    o = np.array(out).mean(0)
    print(f'  p={p:.2f}: 1x mean month {o[0]:+.1%} | Kelly lev {o[1]:6.1f}x | Kelly growth/month {o[2]:+.0%}')
