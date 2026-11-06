"""Support/resistance strength at a price, using only data known before the setup.

- fib: price sits on a wave-1 retracement level (0.382 / 0.5 / 0.618 / 0.786)
- vp:  volume profile (매물대) — lots of volume traded near this price in the
       lookback window before wave 1 topped
- sr:  an older swing high/low sits near this price (old resistance -> support)
"""
import numpy as np

FIBS = (0.382, 0.5, 0.618, 0.786)


def fib_hit(price, start, end, tol):
    for f in FIBS:
        lvl = end - f * (end - start)
        if abs(price / lvl - 1) <= tol:
            return 1
    return 0


def vp_strength(h, l, v, i0, i1, price, tol, bins=200):
    """Volume within +-tol of price, relative to the average band of the same
    width across the window [i0, i1). >1 means heavier than usual."""
    if i1 - i0 < 50:
        return 0.0
    hh, ll, vv = h[i0:i1], l[i0:i1], v[i0:i1]
    lo, hi = ll.min(), hh.max()
    if not (lo < price < hi):
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    mid = (hh + ll) / 2
    hist, _ = np.histogram(mid, bins=edges, weights=vv)
    centers = (edges[:-1] + edges[1:]) / 2
    band = np.abs(centers / price - 1) <= tol
    if not band.any():
        return 0.0
    width = band.sum()
    avg = hist.sum() / bins * width
    return hist[band].sum() / avg if avg > 0 else 0.0


def sr_hit(pivots, before_idx, price, tol, lookback=60):
    """Older swing points (known before `before_idx`) within tol of price."""
    old = [p for p in pivots if p[3] < before_idx][-lookback:]
    return int(sum(abs(p[1] / price - 1) <= tol for p in old) >= 1)
