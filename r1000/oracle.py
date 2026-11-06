"""Spec sheet: how accurate must a robot be to compound 1000%/month (x11) on BTC?
Noisy oracle on TEST (2023-01..2026-10): at each decision bar the robot is in the
right direction with probability p, holds `H` hours, pays OKX taker+slippage
(0.07%/side) and funding. Kelly-optimal leverage, with intrabar liquidation ignored
(so this is an UPPER bound)."""
import numpy as np, pandas as pd
df = pd.read_csv('data/btcusdt_1h.csv', parse_dates=['time']).set_index('time')
c = df.c[df.index >= '2023-01-01']
COST, FUND = 0.0007, 0.0001 / 8
need = np.log(11)                             # log-growth per month for x11
rng = np.random.default_rng(1)
print(f"{'H':>4} {'p':>5} {'edge/trade':>10} {'Kelly lev':>9} {'month x (Kelly)':>15} {'max DD@Kelly':>12}")
for H in [1, 4, 24]:
    rr = c.iloc[::H].pct_change().dropna().values
    for p in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]:
        res = []
        for _ in range(30):
            sgn = np.where(rng.random(len(rr)) < p, np.sign(rr), -np.sign(rr))
            x = sgn * rr - COST * np.abs(np.diff(np.r_[0, sgn])) - FUND * H
            mu, var = x.mean(), x.var()
            f = max(mu / var, 0)
            g = (np.log1p(np.clip(f * x, -0.999, None))).mean() * (24 * 30 / H)
            eq = np.cumprod(1 + np.clip(f * x, -1, None))
            dd = 1 - (eq / np.maximum.accumulate(eq)).min()
            res.append((mu, f, np.exp(g), dd))
        mu, f, gx, dd = np.median(res, 0)
        flag = '  <== x11' if gx >= 11 else ''
        print(f"{H:>4} {p:>5.2f} {mu:>+10.4%} {f:>9.1f} {gx:>15.3g} {dd:>12.0%}{flag}")
