"""Backtest count-based trade setups (the auto-count-platform way) on BTC.

Uses the v4 top-down counter. When the best count says a wave just ended that
should be followed by a trend leg, take the trade:
  IMP/LD after 2 -> wave 3: target 1.618 x wave 1 from the end of 2
  IMP after 4     -> wave 5: target = wave 1 length from the end of 4
  ZZ/FL after B, DZ/TZ/DT/TT after X -> C / Y: target = A (W) length from the end of B (X)
  TRI after E     -> thrust: target = widest part (leg a) from the end of e
Entry at the close that confirmed the swing. Stop variants:
  invalid — the level that kills the count (start of 1, end of 1 for wave 5,
            start of A for C, end of e for a thrust)
  swing   — just beyond the swing that was just made (the prior low/high)
Exit: half at the prior extreme (end of 1 / 3 / A / W), stop to entry, rest at
the Fibonacci target. Fees 0.035% a side (taker after rebate), in R.
"""
import sys, numpy as np, pandas as pd
from ew.v4 import Deg, TFS, PATTERNS, NAMES, check
from ew.v4_run import parent_lookup
from ew.engine import load_1m

FEE, BUF = 0.00035, 0.0005


def setup(pat, k, P, s):
    """-> (direction, invalid level, swing level, tp1, tp2) or None."""
    n = PATTERNS[pat]
    if pat in ('IMP', 'LD') and k == 2:
        w1 = abs(P[1] - P[0]); return s, P[0], P[2], P[1], P[2] + s * 1.618 * w1
    if pat == 'IMP' and k == 4:
        w1 = abs(P[1] - P[0]); return s, P[1], P[4], P[3], P[4] + s * w1
    if pat in ('ZZ', 'FL') and k == 2:
        A = abs(P[1] - P[0]); return s, P[0], P[2], P[1], P[2] + s * A
    if pat in ('DZ', 'TZ', 'DT', 'TT') and k % 2 == 0 and k < n:
        W = abs(P[1] - P[0]); return s, P[0], P[k], P[k - 1], P[k] + s * W
    if pat == 'TRI' and k == 5:
        a = abs(P[1] - P[0]); return -s, P[5], P[5], P[4], P[5] - s * a
    return None


def trade(D, k0, px, d, stop, tp1, tp2, maxbars=5000):
    dist = d * (px - stop) / px
    if dist <= 0 or d * (tp1 - px) <= 0 or d * (tp2 - tp1) <= 0: return None
    unit = dist + 2 * FEE; h, l = D.h, D.l; half = False; st = stop
    for t in range(k0, min(len(h), k0 + maxbars)):
        if (l[t] <= st) if d > 0 else (h[t] >= st):
            leg = d * (st - px) / px - 2 * FEE
            return ((0.5 * (d * (tp1 - px) / px - 2 * FEE) + 0.5 * leg) if half else leg) / unit, dist, t - k0
        if not half and ((h[t] >= tp1) if d > 0 else (l[t] <= tp1)):
            half = True; st = px * (1 + d * 2 * FEE); continue
        if half and ((h[t] >= tp2) if d > 0 else (l[t] <= tp2)):
            return (0.5 * (d * (tp1 - px) / px - 2 * FEE) + 0.5 * (d * (tp2 - px) / px - 2 * FEE)) / unit, dist, t - k0
    return None


def run(tfs):
    m1 = load_1m(); rows = []; parent_tl = None
    for tf in tfs:
        D = Deg(tf, m1); tl_t, tl_pos, tl_dir = [], [], []; busy = {'invalid': -1, 'swing': -1}
        for j in range(1, len(D.p)):
            now = D.kn[j]; ppos, pdir = parent_lookup(parent_tl, now)
            k_now = np.searchsorted(D.ct, now, side='left')
            if k_now + 1 >= len(D.c): break
            px = D.c[k_now]; best = None
            for pat, n in PATTERNS.items():
                for k in range(1, n + 1):
                    st = j - k
                    if st < 0: continue
                    P = list(D.p[st:j + 1]); s = 1 if P[1] > P[0] else -1
                    S = [D.structure(st + q, st + q + 1) for q in range(k)]
                    ok, sc, why, info = check(pat, P, s, S, D, D.i[st:j + 1], ppos)
                    if not ok: continue
                    wave = ppos.split(':')[-1] if ppos != 'none' else ''
                    motive_parent = wave in ('1', '3', '5', 'A', 'C', 'W', 'Y', 'Z'); builds_motive = pat in ('IMP', 'LD', 'ED')
                    bonus = 0.0
                    if pdir != 0 and wave:
                        if motive_parent and builds_motive and s == pdir: bonus = 0.3
                        elif not motive_parent and not builds_motive and s == -pdir: bonus = 0.3
                        elif (builds_motive and s == -pdir) or (not builds_motive and s == pdir): bonus = -0.2
                    tot = sc + bonus + 0.04 * k
                    if best is None or tot > best[0]: best = (tot, pat, k, s, P, bonus, sc)
            if best is None: continue
            tot, pat, k, s, P, bonus, sc = best; n = PATTERNS[pat]
            tl_t.append(now); tl_pos.append(f'{pat}:{NAMES[pat][k] if k < n else "post"}'); tl_dir.append(s if k % 2 == 0 or k == n else -s)
            su = setup(pat, k, P, s)
            if su is None: continue
            d, inv, sw, tp1, tp2 = su
            for mode, lvl in (('invalid', inv), ('swing', sw)):
                if k_now + 1 <= busy[mode]: continue
                stop = lvl * (1 - d * BUF)
                r = trade(D, k_now + 1, px, d, stop, tp1, tp2)
                if r is None: continue
                R, dist, bars = r; busy[mode] = k_now + 1 + bars
                rows.append((tf, pd.Timestamp(now, tz='UTC'), mode, pat, NAMES[pat][k - 1], d, bonus, sc, R, dist, bars,
                             abs(tp2 - px) / abs(px - stop)))
        parent_tl = (np.array(tl_t), tl_pos, tl_dir)
        print(tf, 'trades', sum(1 for r in rows if r[0] == tf), flush=True)
    return pd.DataFrame(rows, columns=['tf', 'time', 'stop', 'pattern', 'wave', 'side', 'bonus', 'score', 'R', 'stop_dist', 'bars', 'rr2'])


if __name__ == '__main__':
    tfs = sys.argv[1:] or TFS
    T = run(tfs); T.to_csv('ew/v4_trades_btc.csv.gz', index=False)
    pd.set_option('display.width', 250); days = (T.time.max() - T.time.min()).days; out = []
    g = T.groupby(['stop', 'tf']).agg(n=('R', 'size'), win=('R', lambda x: (x > 0).mean()), avgR=('R', 'mean'),
                                      med_stop=('stop_dist', 'median'), med_rr_target=('rr2', 'median'))
    g['per_day'] = g.n / days; out.append('ALL SETUPS\n' + g.round(3).to_string())
    out.append('\nBY SETUP (stop = swing)\n' + T[T.stop == 'swing'].groupby(['pattern', 'wave']).agg(
        n=('R', 'size'), win=('R', lambda x: (x > 0).mean()), avgR=('R', 'mean')).round(3).to_string())
    out.append('\nBY SETUP (stop = invalid)\n' + T[T.stop == 'invalid'].groupby(['pattern', 'wave']).agg(
        n=('R', 'size'), win=('R', lambda x: (x > 0).mean()), avgR=('R', 'mean')).round(3).to_string())
    out.append('\nTOP-DOWN AGREEMENT\n' + T.groupby(['stop', 'bonus']).R.agg(['size', 'mean']).round(3).to_string())
    out.append('\nBY YEAR (agreeing counts, stop = swing)\n' + T[(T.stop == 'swing') & (T.bonus > 0)].groupby(T.time.dt.year).R.agg(['size', 'mean']).round(3).to_string())
    txt = '\n'.join(out); print(txt); open('logs/ew_v4_trades_report.txt', 'w').write(txt)
