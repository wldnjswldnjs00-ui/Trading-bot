"""Scan every degree, label each completed pattern, and record what happened next.

For each pattern the "question" the engine asks itself is binary and is settled
by price alone (first level touched on that degree's bars after the pattern
became known):
  impulse / diagonal  "is the 5-wave move really over?"
                      yes = price comes back to the wave-4 end before a new extreme
  zigzag / flat       "is the correction over and the trend resuming?"
                      yes = price goes beyond the correction start before beyond its end
  triangle            "does the thrust go the expected way?"
                      yes = price breaks the d end before the e end
Each row keeps the levels, so a trade "bet on yes, stop at the no level" has a
known reward:risk at the moment of decision.
"""
import numpy as np, pandas as pd
from numba import njit
from ew.engine import TFS, build_degrees, classify, subwaves, parent_dir, load_1m


@njit(cache=True)
def first_touch(h, l, i0, up_lvl, dn_lvl, maxbars):
    """1 = up level first, -1 = down level first, 0 = both in one bar / none."""
    n = len(h)
    for i in range(i0, min(n, i0 + maxbars)):
        u = h[i] >= up_lvl; d = l[i] <= dn_lvl
        if u and d: return 0, i - i0
        if u: return 1, i - i0
        if d: return -1, i - i0
    return 0, -1


def scan(degs, maxbars=2000):
    rows = []
    for g, tf in enumerate(TFS):
        D = degs[tf]; up_deg = degs[TFS[g - 1]] if g > 0 else None; lo_deg = degs[TFS[g + 1]] if g + 1 < len(TFS) else None
        b = D.bars; bt = b.close_time.values.astype('datetime64[ns]').astype(np.int64)
        h, l, c = b.h.values, b.l.values, b.c.values
        for j in range(3, len(D.p)):
            now = D.known[j]
            k_now = np.searchsorted(bt, now, side='left')                # bar whose close made it known
            if k_now + 1 >= len(c): continue
            px_now = c[k_now]
            for n in (4, 6):
                if j - n + 1 < 0: continue
                sl = slice(j - n + 1, j + 1); P = D.p[sl]; T = D.t[sl]; kinds = D.kind[sl]
                up = kinds[0] == -1
                sub = [subwaves(lo_deg, T[i], T[i + 1], now) for i in range(n - 1)]
                for pat, score, info in classify(list(P), up, sub):
                    s = 1 if up else -1
                    if pat.startswith(('impulse', 'diagonal')):
                        yes, no = P[4], P[5]                              # back to wave-4 end vs new extreme
                        q = 'move_over'
                    elif pat.startswith(('zigzag', 'flat')):
                        yes, no = P[0], P[3]                              # beyond correction start vs beyond its end
                        q = 'trend_resumes'
                    else:
                        yes, no = P[4], P[5]                              # d end vs e end
                        q = 'thrust'
                    # yes is on the -s side for every pattern above
                    if (s > 0 and not (yes < px_now < no)) or (s < 0 and not (no < px_now < yes)):
                        continue                                          # already settled when it became known
                    up_lvl, dn_lvl = (no, yes) if s > 0 else (yes, no)
                    res, bars = first_touch(h, l, k_now + 1, up_lvl, dn_lvl, maxbars)
                    if res == 0: continue
                    ok = (res == -1) if s > 0 else (res == 1)
                    rr = abs(yes - px_now) / abs(no - px_now)
                    pdir = parent_dir(up_deg, now)
                    pat_dir = s if not pat.startswith(('zigzag', 'flat')) else -s   # trend direction the pattern implies
                    rows.append(dict(tf=tf, pattern=pat, question=q, time=pd.Timestamp(now, tz='UTC'), dir=s,
                                     ctx='with' if pdir == s else 'against' if pdir == -s else 'none',
                                     score=score, subs_known=all(x is not None for x in sub), ok=bool(ok), rr=rr,
                                     bars=bars, shape=info.get('shape', ''), ext=info.get('ext', 0)))
        print(tf, 'swings', len(D.p), 'patterns so far', len(rows), flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    m1 = load_1m(); degs = build_degrees(m1)
    R = scan(degs); R.to_csv('ew/patterns_btc.csv.gz', index=False)
    print(R.groupby(['tf', 'pattern']).ok.agg(['size', 'mean']).round(3).to_string())
