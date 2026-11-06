"""Nested top-down entry: big wave 2 -> its end on the middle degree -> new
wave 1-2 on the small degree -> enter, stop tiny, target the big wave 3.

  BIG    best count says wave 1 (IMP/LD) or A (ZZ) just finished: a big pullback
         (wave 2 / B) is now in progress. Big invalidation = start of wave 1 / A.
  MID    while that pullback is live, the middle degree's best count completes a
         correction against the big direction (ZZ/FL/DZ/DT C or Y, TRI e, or any
         complete corrective count), with price still above the big start.
         -> treat as the end of big wave 2.
  SMALL  after that, the small degree's best count shows wave 1-2 (IMP/LD at k=2)
         in the big direction -> enter at the close that confirmed it.
         Stop just beyond the small wave-2 swing.
  EXIT   half at the end of big wave 1 (prior high), stop to entry, rest at the
         big wave-3 target: end of big 2 + 1.618 x big wave 1.
All on the small degree's bars. Fees 0.035% a side. R includes fees.
"""
import sys, numpy as np, pandas as pd
from ew.v4 import Deg, PATTERNS, NAMES, check
from ew.v4_run import parent_lookup
from ew.engine import load_1m

FEE, BUF = 0.00035, 0.0005
ALL = '--all' in sys.argv          # use every live count, not only the best one
CORR_DONE = {('ZZ', 3), ('FL', 3), ('DZ', 3), ('DT', 3), ('TZ', 5), ('TT', 5), ('TRI', 5)}


def best_counts(D, parent_tl):
    """Best count at every swing of degree D: list of (now, pat, k, s, P)."""
    out = []; tl_t, tl_pos, tl_dir = [], [], []
    for j in range(1, len(D.p)):
        now = D.kn[j]; ppos, pdir = parent_lookup(parent_tl, now); best = None
        for pat, n in PATTERNS.items():
            for k in range(1, n + 1):
                st = j - k
                if st < 0: continue
                P = list(D.p[st:j + 1]); s = 1 if P[1] > P[0] else -1
                S = [D.structure(st + q, st + q + 1) for q in range(k)]
                ok, sc, _, _ = check(pat, P, s, S, D, D.i[st:j + 1], ppos)
                if not ok: continue
                wave = ppos.split(':')[-1] if ppos != 'none' else ''
                mp = wave in ('1', '3', '5', 'A', 'C', 'W', 'Y', 'Z'); bm = pat in ('IMP', 'LD', 'ED'); bonus = 0.0
                if pdir != 0 and wave:
                    if mp and bm and s == pdir: bonus = 0.3
                    elif not mp and not bm and s == -pdir: bonus = 0.3
                    elif (bm and s == -pdir) or (not bm and s == pdir): bonus = -0.2
                tot = sc + bonus + 0.04 * k
                if best is None or tot > best[0]: best = (tot, pat, k, s, P)
                if ALL: out.append((now, pat, k, s, P))
        if best is None: continue
        _, pat, k, s, P = best; n = PATTERNS[pat]
        if not ALL: out.append((now, pat, k, s, P))
        tl_t.append(now); tl_pos.append(f'{pat}:{NAMES[pat][k] if k < n else "post"}'); tl_dir.append(s if k % 2 == 0 or k == n else -s)
    return out, (np.array(tl_t), tl_pos, tl_dir)


def simulate(Sm, k0, d, px, stop, tp1, tp2, maxbars=40000):
    dist = d * (px - stop) / px
    if dist <= 0 or d * (tp1 - px) <= 0 or d * (tp2 - tp1) <= 0: return None
    unit = dist + 2 * FEE; half = False; st = stop; h, l = Sm.h, Sm.l
    for t in range(k0, min(len(h), k0 + maxbars)):
        if (l[t] <= st) if d > 0 else (h[t] >= st):
            leg = d * (st - px) / px - 2 * FEE
            return ((0.5 * (d * (tp1 - px) / px - 2 * FEE) + 0.5 * leg) if half else leg) / unit, dist, t - k0, half
        if not half and ((h[t] >= tp1) if d > 0 else (l[t] <= tp1)):
            half = True; st = px * (1 + d * 2 * FEE); continue
        if half and ((h[t] >= tp2) if d > 0 else (l[t] <= tp2)):
            return (0.5 * (d * (tp1 - px) / px - 2 * FEE) + 0.5 * (d * (tp2 - px) / px - 2 * FEE)) / unit, dist, t - k0, True
    return None


def run(big, mid, small, m1):
    B = Deg(big, m1); M = Deg(mid, m1); Sm = Deg(small, m1)
    bc, btl = best_counts(B, None); mc, mtl = best_counts(M, btl); sc, _ = best_counts(Sm, mtl)
    mt = np.array([x[0] for x in mc]); st_ = np.array([x[0] for x in sc]); bt = np.array([x[0] for x in bc])
    bswing = np.unique(bt)
    rows = []; busy_until = -1; busy_until_t = -1
    for bi, (now, pat, k, s, P) in enumerate(bc):
        if not ((pat in ('IMP', 'LD') and k == 1) or (pat == 'ZZ' and k == 1)): continue
        w1 = abs(P[1] - P[0]); start, end1 = P[0], P[1]
        nx = np.searchsorted(bswing, now, side='right')
        valid_until = bswing[nx] if nx < len(bswing) else np.iinfo(np.int64).max
        if now <= busy_until_t: continue
        # look for the middle-degree correction that ends big wave 2
        a = np.searchsorted(mt, now, side='right'); b = np.searchsorted(mt, valid_until + 30 * 86400 * 10**9, side='right')
        end2 = None
        for q in range(a, b):
            t2, p2, k2, s2, P2 = mc[q]
            if s2 != -s: continue
            done = (p2, k2) in CORR_DONE or (p2 in ('IMP', 'LD') and k2 == 5)
            if not done: continue
            low = P2[-1]
            if s * (low - start) <= 0: break                           # big wave 2 went past the start: count dead
            if s * (end1 - low) <= 0: continue
            end2 = (t2, low); break
        if end2 is None: continue
        t2, low = end2
        # small degree: first wave 1-2 in the big direction after that
        a = np.searchsorted(st_, t2, side='right')
        for q in range(a, min(len(sc), a + 400)):
            t3, p3, k3, s3, P3 = sc[q]
            if s3 == s and p3 in ('IMP', 'LD') and k3 == 2:
                if s * (P3[2] - low) < 0: break                          # small wave 2 undercut the big wave-2 end
                k_now = np.searchsorted(Sm.ct, t3, side='left')
                if k_now + 1 >= len(Sm.c) or k_now + 1 <= busy_until: break
                px = Sm.c[k_now]; stop = P3[2] * (1 - s * BUF)
                tp1 = end1; tp2 = low + s * 1.618 * w1
                r = simulate(Sm, k_now + 1, s, px, stop, tp1, tp2)
                if r is None: break
                R, dist, bars, hit1 = r; busy_until = k_now + 1 + bars; busy_until_t = Sm.ct[min(len(Sm.ct) - 1, busy_until)]
                rows.append(dict(combo=f'{big}>{mid}>{small}', time=pd.Timestamp(t3, tz='UTC'), side=s, big_pat=pat, R=R,
                                 stop_dist=dist, hit_tp1=hit1, rr_tp2=abs(tp2 - px) / abs(px - stop), hours=bars * Sm.ct[1:2].item() * 0))
                break
            if s3 == -s and p3 in ('IMP', 'LD') and k3 >= 3: break     # small degree turned against: skip this one
    return pd.DataFrame(rows)


if __name__ == '__main__':
    combos = [('1D', '1h', '5min'), ('4h', '15min', '5min'), ('4h', '1h', '5min')]
    m1 = load_1m(); out = []
    for c in combos:
        T = run(*c, m1); out.append(T); print(c, len(T), flush=True)
    T = pd.concat(out); T.to_csv('ew/nested_trades_btc%s.csv.gz' % ('_all' if ALL else ''), index=False)
    pd.set_option('display.width', 220)
    g = T.groupby('combo').agg(n=('R', 'size'), win=('R', lambda x: (x > 0).mean()), hit_tp1=('hit_tp1', 'mean'), avgR=('R', 'mean'),
                               medR=('R', 'median'), med_stop=('stop_dist', 'median'), med_rr=('rr_tp2', 'median'), best=('R', 'max'))
    yr = T.groupby(['combo', T.time.dt.year]).R.agg(['size', 'mean']).unstack('combo')
    txt = 'NESTED TOP-DOWN ENTRIES (BTC 2017-2026, fees included)\n' + g.round(3).to_string() + '\n\nBY YEAR\n' + yr.round(3).to_string()
    print(txt); open('logs/ew_nested%s_report.txt' % ('_all' if ALL else ''), 'w').write(txt)
