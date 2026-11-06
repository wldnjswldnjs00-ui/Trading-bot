"""Elliott wave engine v1 — causal (no look-ahead).

Every pivot carries the bar index at which it became *known*. Any decision at
bar t only uses pivots known at or before t, so the backtest sees exactly what
a live bot would have seen.
"""
import numpy as np
import pandas as pd


def atr(df, n=100):
    h, l, c = df.h.values, df.l.values, df.c.values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


def zigzag(df, k, a):
    """Causal zigzag. A swing is confirmed once price moves k*ATR against it.
    Returns list of (idx, price, kind, known_at) with kind +1 high / -1 low."""
    h, l = df.h.values, df.l.values
    piv = []
    d = 0
    ext_i, ext_p = 0, h[0]
    lo_i, lo_p, hi_i, hi_p = 0, l[0], 0, h[0]
    for t in range(1, len(h)):
        th = k * a[t]
        if d == 0:
            if h[t] > hi_p: hi_i, hi_p = t, h[t]
            if l[t] < lo_p: lo_i, lo_p = t, l[t]
            if hi_p - lo_p > th:
                if hi_i > lo_i:
                    piv.append((lo_i, lo_p, -1, t)); d = 1; ext_i, ext_p = hi_i, hi_p
                else:
                    piv.append((hi_i, hi_p, 1, t)); d = -1; ext_i, ext_p = lo_i, lo_p
            continue
        if d == 1:
            if h[t] >= ext_p: ext_i, ext_p = t, h[t]
            elif ext_p - l[t] > th:
                piv.append((ext_i, ext_p, 1, t)); d = -1; ext_i, ext_p = t, l[t]
        else:
            if l[t] <= ext_p: ext_i, ext_p = t, l[t]
            elif h[t] - ext_p > th:
                piv.append((ext_i, ext_p, -1, t)); d = 1; ext_i, ext_p = t, h[t]
    return piv


def legs_between(piv, i0, i1):
    """Lower-degree pivots strictly inside (i0, i1)."""
    return [p for p in piv if i0 < p[0] < i1]


def is_motive(pts, up):
    """pts = 6 prices p0..p5. Checks impulse rules; returns 'impulse',
    'diagonal' or None."""
    s = 1 if up else -1
    p = [s * x for x in pts]
    w1, w3, w5 = p[1] - p[0], p[3] - p[2], p[5] - p[4]
    if min(w1, w3, w5) <= 0: return None
    if p[2] <= p[0]: return None                     # wave 2 never beyond start of 1
    if w3 < w1 and w3 < w5: return None               # wave 3 never shortest
    if p[4] > p[1]: return 'impulse'                  # no overlap 4 vs 1
    # overlap allowed only for diagonals: wedge shape (contracting or expanding)
    if p[4] > p[0] and ((w3 < w1 and w5 < w3) or (w3 > w1 and w5 > w3)):
        return 'diagonal'
    return None


def internal_count(lower, i0, i1, up, want):
    """Does the leg from i0 to i1 subdivide into `want` (5 or 3) waves at the
    lower degree? Returns True / False / None (not enough lower pivots)."""
    inner = legs_between(lower, i0, i1)
    segs = len(inner) + 1
    if segs < want: return False
    if want == 5 and segs >= 5:
        return True
    return segs >= 3


def fib_score(r, targets=(0.382, 0.5, 0.618, 0.786)):
    return max(0.0, 1 - min(abs(r - x) for x in targets) / 0.1)
