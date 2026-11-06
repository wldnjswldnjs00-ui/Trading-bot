"""How do people on X show +thousands %/week with leveraged bots?
Simulate a crowd of simple Claude-Code-style bots on real BTC 1h (2023-01..2026-10):
  signal = follow (or fade) the last k-hour move, k in 1..48
  leverage in {10,20,50,100}, all-in isolated margin, compounding within the week,
  OKX taker+slippage 0.07%/side on notional, funding, intrabar liquidation.
Each bot restarts every week (like a fresh screenshot). Then: who hits 1000%+,
and do the same bots do it again the next week?"""
import numpy as np, pandas as pd
df = pd.read_csv('data/btcusdt_1h.csv', parse_dates=['time']).set_index('time')
df = df[df.index >= '2023-01-01']
c, h, l = df.c.values, df.h.values, df.l.values
r = np.r_[0, c[1:] / c[:-1] - 1]
up = np.r_[0, h[1:] / c[:-1] - 1]; dn = np.r_[0, l[1:] / c[:-1] - 1]
COST, FUND = 0.0007, 0.0001 / 8
week = df.index.to_period('W').astype(str).values
bots = [(k, d, L) for k in range(1, 49) for d in (1, -1) for L in (10, 20, 50, 100)]
res = np.zeros((len(bots), len(np.unique(week))))
weeks = pd.unique(week)
for j, (k, d, L) in enumerate(bots):
    sig = d * np.sign(np.r_[np.zeros(k), c[k:] / c[:-k] - 1])
    pos = np.r_[0, sig[:-1]]                                # decided at previous close
    adv = np.where(pos > 0, dn, -up) * np.abs(pos)
    bar = L * (pos * r - COST * np.abs(np.diff(np.r_[0, pos])) - FUND * np.abs(pos))
    for wi, w in enumerate(weeks):
        m = week == w
        if (L * adv[m] <= -0.99).any():
            first = np.argmax(L * adv[m] <= -0.99)
            eq = np.prod(np.clip(1 + bar[m][:first], 0, None))
            eq = 0.0
        else:
            eq = np.prod(np.clip(1 + bar[m], 0, None))
        res[j, wi] = eq - 1
R = res.ravel()
print(f'bots: {len(bots)}  weeks: {len(weeks)}  bot-weeks: {R.size:,}')
print(f'liquidated (-100%)      : {(R <= -0.99).mean():.1%}')
print(f'median week             : {np.median(R):+.1%}')
for t in [1, 3, 10, 30]:
    print(f'weeks >= +{t*100:>5.0f}%       : {(R >= t).sum():>6,}  ({(R >= t).mean():.3%})')
print(f'best single week        : {R.max():+,.0%}')
by_L = pd.DataFrame({'L': np.repeat([b[2] for b in bots], len(weeks)), 'R': R})
print('\nby leverage:  liquidated | median | share of weeks >= +1000%')
for L, g in by_L.groupby('L'):
    print(f'  {L:>3}x   {(g.R <= -0.99).mean():>6.1%}   {g.R.median():>+7.1%}   {(g.R >= 10).mean():.3%}')
# persistence: bots that made >=1000% in week i — what happens to them in week i+1?
nxt = [res[j, i + 1] for j in range(len(bots)) for i in range(len(weeks) - 1) if res[j, i] >= 10]
nxt = np.array(nxt)
print(f'\nafter a +1000% week ({len(nxt)} cases), NEXT week:')
print(f'  liquidated {(nxt <= -0.99).mean():.0%} | median {np.median(nxt):+.0%} | again >= +1000%: {(nxt >= 10).mean():.1%}')
# whole-period: compounding a bot that never resets
life = (1 + res).prod(1)
print(f'\nsame bots compounding the whole 3.8 years: alive {(life > 0.01).mean():.1%}, best final x {life.max():.3g}')
