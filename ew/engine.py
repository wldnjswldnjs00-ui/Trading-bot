"""Multi-degree Elliott wave engine (causal).

Degrees = timeframes, biggest first: 1W, 1D, 4h, 1h, 15m, 5m, 1m. Each degree's
swings come from a zigzag that only confirms a swing once price has turned by
K x ATR of that timeframe, and every swing carries the moment it became known.
Nothing ever looks past "now".

At the moment a new swing is confirmed on a degree, the last swings are tested
against the wave patterns. Hard rules reject; guidelines score; the degree
below tells us how many sub-waves each leg has (5 for motive, 3 for corrective).
The degree above gives the context (is this move with or against the bigger
leg) — that is the top-down part.
"""
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from numba import njit

TFS = ['1W', '1D', '4h', '1h', '15min', '5min', '1min']          # big -> small
K = 3.0                                                              # zigzag size in ATRs, every degree


def load_1m(path='data/btc_spot_1m.npy'):
    D = np.load(path)
    df = pd.DataFrame(D[:, 1:], columns=['o', 'h', 'l', 'c', 'v', 'tb'])
    df.index = pd.to_datetime(D[:, 0].astype(np.int64), unit='ms', utc=True)
    return df


def resample(m1, tf):
    if tf == '1min': d = m1
    else:
        rule = 'W-MON' if tf == '1W' else tf
        kw = dict(label='left', closed='left') if tf == '1W' else dict(origin='epoch')
        d = m1.resample(rule, **kw).agg(dict(o='first', h='max', l='min', c='last', v='sum', tb='sum')).dropna()
    out = d.reset_index().rename(columns={'index': 'time'})
    step = pd.Timedelta('7D') if tf == '1W' else pd.Timedelta(tf)
    out['close_time'] = out.time + step                              # bar info is usable from here
    return out


def _atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


@njit(cache=True)
def _zigzag(h, l, a, k):
    n = len(h); idx = np.empty(n, np.int64); px = np.empty(n); kind = np.empty(n, np.int8); known = np.empty(n, np.int64); m = 0
    d = 0; ei = 0; ep = h[0]; li = 0; lp = l[0]; hi = 0; hp = h[0]
    for t in range(1, n):
        th = k * a[t]
        if d == 0:
            if h[t] > hp: hi = t; hp = h[t]
            if l[t] < lp: li = t; lp = l[t]
            if hp - lp > th:
                if hi > li:
                    idx[m] = li; px[m] = lp; kind[m] = -1; known[m] = t; m += 1; d = 1; ei = hi; ep = hp
                else:
                    idx[m] = hi; px[m] = hp; kind[m] = 1; known[m] = t; m += 1; d = -1; ei = li; ep = lp
            continue
        if d == 1:
            if h[t] >= ep: ei = t; ep = h[t]
            elif ep - l[t] > th:
                idx[m] = ei; px[m] = ep; kind[m] = 1; known[m] = t; m += 1; d = -1; ei = t; ep = l[t]
        else:
            if l[t] <= ep: ei = t; ep = l[t]
            elif h[t] - ep > th:
                idx[m] = ei; px[m] = ep; kind[m] = -1; known[m] = t; m += 1; d = 1; ei = t; ep = h[t]
    return idx[:m], px[:m], kind[:m], known[:m]


@dataclass
class Degree:
    tf: str
    bars: pd.DataFrame
    t: np.ndarray            # swing bar time (ns)
    p: np.ndarray            # swing price
    kind: np.ndarray         # +1 high / -1 low
    known: np.ndarray        # time the swing became known (ns)


def build_degrees(m1, tfs=TFS, k=K):
    out = {}
    for tf in tfs:
        b = resample(m1, tf)
        h, l, c = b.h.values, b.l.values, b.c.values
        a = _atr(h, l, c, 14)
        i, p, kd, kn = _zigzag(h, l, a, k)
        tt = b.time.values.astype('datetime64[ns]').astype(np.int64)
        ct = b.close_time.values.astype('datetime64[ns]').astype(np.int64)
        out[tf] = Degree(tf, b, tt[i], p, kd, ct[kn])
    return out


def subwaves(lower, t0, t1, now):
    """Sub-waves of the leg between swing times t0..t1 at the lower degree
    (swings known by `now`). None when the lower degree is too coarse to tell."""
    if lower is None: return None
    a = np.searchsorted(lower.t, t0, side='right'); b = np.searchsorted(lower.t, t1, side='left')
    inner = lower.known[a:b]
    if len(inner) and inner.max() > now: inner = inner[inner <= now]
    n = len(inner) + 1
    return n if n >= 3 else None


def parent_dir(upper, now):
    """Direction of the bigger leg in progress at `now`: after a confirmed high
    the bigger leg is down (-1), after a low it is up (+1)."""
    if upper is None: return 0
    j = np.searchsorted(upper.known, now, side='right') - 1
    return 0 if j < 0 else -int(upper.kind[j])


FIB = np.array([0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618])


def near_fib(x, tol=0.06):
    return float(np.min(np.abs(FIB - x)) <= tol)


def classify(P, up, sub):
    """P: swing prices (oldest first). up: direction of the first leg.
    sub: sub-wave counts per leg (None = unknown). Returns list of (pattern, score, info)."""
    s = 1 if up else -1; out = []
    L = [s * (P[i + 1] - P[i]) for i in range(len(P) - 1)]           # signed leg sizes, motive legs > 0
    if len(P) == 6:
        w1, w2, w3, w4, w5 = L[0], -L[1], L[2], -L[3], L[4]
        if min(w1, w2, w3, w4, w5) > 0 and w2 < w1 and not (w3 < w1 and w3 < w5):
            overlap = s * (P[4] - P[1]) <= 0
            sc = []
            sc.append(near_fib(w2 / w1)); sc.append(near_fib(w4 / w3))
            sc.append(near_fib(w3 / w1) or near_fib(w5 / w1) or near_fib(w5 / w3))
            sc.append(float((w2 / w1 > 0.5) != (w4 / w3 > 0.5)))       # alternation (deep vs shallow)
            if not overlap:
                subs = [sub[i] for i in range(5)]
                if all(x is not None for x in subs):
                    sc.append(float(subs[0] >= 5 and subs[2] >= 5 and subs[4] >= 5))
                out.append(('impulse', float(np.mean(sc)), dict(ext=int(np.argmax([w1, w3, w5])) * 2 + 1)))
            else:
                contracting = w3 < w1 and w5 < w3 and w4 < w2
                expanding = w3 > w1 and w5 > w3 and w4 > w2
                if (contracting or expanding) and s * (P[4] - P[0]) > 0:
                    kind = 'leading' if (sub[0] or 0) >= 5 and (sub[2] or 0) >= 5 else 'ending' if sub[0] is not None and sub[0] < 5 else 'diag'
                    out.append((f'diagonal_{kind}', float(np.mean(sc[:3])), dict(shape='contracting' if contracting else 'expanding')))
        # triangle a-e: legs alternate, each smaller (contracting) or bigger (expanding) than two back
        a, b, c, d, e = [abs(x) for x in L]
        if L[0] * L[1] < 0 and b < a * 1.0 and c < a and d < b and e < c and e < d * 1.0 and b > 0.3 * a:
            out.append(('triangle_contracting', float(np.mean([near_fib(c / a), near_fib(e / c)])), {}))
        elif c > a and d > b and e > c:
            out.append(('triangle_expanding', 0.5, {}))
    if len(P) == 4:
        A, B, C = L[0], -L[1], L[2]
        if A > 0 and B > 0 and C > 0:
            r, q = B / A, C / A
            subA = sub[0]
            if 0.236 <= r <= 0.886 and s * (P[3] - P[1]) > 0 and 0.618 <= q <= 2.618:
                sc = [near_fib(r), near_fib(q)] + ([float(subA >= 5)] if subA is not None else [])
                out.append(('zigzag', float(np.mean(sc)), {}))
            if 0.886 < r <= 1.382:
                if s * (P[3] - P[1]) > 0:
                    out.append(('flat_expanded' if r > 1.05 else 'flat_regular', float(np.mean([near_fib(q), float(subA is None or subA < 5)])), {}))
                elif q >= 0.5:
                    out.append(('flat_running', 0.5, {}))
    return out
