"""Spec sheet: what does a strategy need so that, in a month,
   (a) equity >= 11x  (+1000%)  and  (b) drawdown inside the month < 10% ?
Monte Carlo over win rate p, reward:risk b, trades per month N, risk per trade r
(fraction of seed lost on a losing trade). 20,000 simulated months each.
Reports the share of months that satisfy BOTH (a) and (b)."""
import numpy as np
rng = np.random.default_rng(0)
def months(p, b, N, r, M=20000):
    w = rng.random((M, N)) < p
    g = np.where(w, 1 + b * r, 1 - r)
    eq = np.cumprod(g, axis=1)
    dd = 1 - (eq / np.maximum.accumulate(np.c_[np.ones(M), eq], axis=1)[:, 1:]).min(axis=1)
    return ((eq[:, -1] >= 11) & (dd < 0.10)).mean(), np.median(eq[:, -1]), np.median(dd)
print(f"{'win':>5} {'R:R':>4} {'trades/mo':>9} {'risk/trade':>10} | {'months OK':>9} {'median x':>9} {'median DD':>9}")
best = []
for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    for b in [1, 2, 3, 5]:
        for N in [30, 60, 120, 240]:
            # largest growth while a losing streak of k stays under 10%: try a few r values
            for r in [0.005, 0.01, 0.02, 0.03, 0.05]:
                ok, mx, dd = months(p, b, N, r, 4000)
                best.append((ok, p, b, N, r, mx, dd))
best.sort(reverse=True)
for ok, p, b, N, r, mx, dd in best[:15]:
    print(f"{p:>5.0%} {b:>4} {N:>9} {r:>10.1%} | {ok:>9.0%} {mx:>9.1f} {dd:>9.1%}")
print('\nminimum combos that reach >= 90% of months OK:')
for ok, p, b, N, r, mx, dd in sorted([x for x in best if x[0] >= 0.9], key=lambda x: (x[1], x[3])):
    print(f"   win {p:.0%}, R:R {b}, {N} trades/month ({N / 30:.0f}/day), risk {r:.1%}/trade -> {ok:.0%} of months OK")
