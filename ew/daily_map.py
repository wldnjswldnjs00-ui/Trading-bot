"""Daily count as the map, day-trade the sub-waves on 1h / 15m / 5m.

The daily best count says which wave is in progress and its direction:
  motive wave in progress (1, 3, 5, A, C, W, Y, Z) going d
      -> only take small-degree impulse setups going d: after wave 2 (-> 3)
         and after wave 4 (-> 5)
  corrective wave in progress (2, 4, B, X) going d
      -> only take small-degree corrective setups going d: after b (-> c),
         after x (-> y)
Everything else is skipped. Targets / stops / fees as in v4_trades.
"""
import sys, numpy as np, pandas as pd
from ew.v4 import Deg, PATTERNS, NAMES, check
from ew.v4_trades import setup, trade, FEE, BUF
from ew.nested import best_counts
from ew.v4_run import parent_lookup
from ew.engine import load_1m

MOTIVE_W = set('135ACWYZ')


def run(child_tfs, m1):
    Dd = Deg('1D', m1); _, dtl = best_counts(Dd, None)
    rows = []
    for tf in child_tfs:
        D = Deg(tf, m1); busy = {'invalid': -1, 'swing': -1}
        for j in range(1, len(D.p)):
            now = D.kn[j]; dpos, ddir = parent_lookup(dtl, now)
            if dpos == 'none' or ddir == 0: continue
            wave = dpos.split(':')[-1]
            if wave == 'post': continue
            want_motive = wave in MOTIVE_W
            k_now = np.searchsorted(D.ct, now, side='left')
            if k_now + 1 >= len(D.c): break
            px = D.c[k_now]; best = None
            for pat, n in PATTERNS.items():
                for k in range(1, n + 1):
                    st = j - k
                    if st < 0: continue
                    P = list(D.p[st:j + 1]); s = 1 if P[1] > P[0] else -1
                    if s != ddir: continue                                   # must go the daily way
                    is_m = pat in ('IMP', 'LD')
                    if want_motive and not (is_m and k in (2, 4)): continue
                    if not want_motive and not (pat in ('ZZ', 'FL', 'DZ', 'DT', 'TZ', 'TT') and k % 2 == 0 and k < n): continue
                    S = [D.structure(st + q, st + q + 1) for q in range(k)]
                    ok, sc, _, _ = check(pat, P, s, S, D, D.i[st:j + 1], dpos)
                    if not ok: continue
                    if best is None or sc + 0.04 * k > best[0]: best = (sc + 0.04 * k, pat, k, s, P)
            if best is None: continue
            _, pat, k, s, P = best; su = setup(pat, k, P, s)
            if su is None: continue
            d, inv, sw, tp1, tp2 = su
            for mode, lvl in (('invalid', inv), ('swing', sw)):
                if k_now + 1 <= busy[mode]: continue
                r = trade(D, k_now + 1, px, d, lvl * (1 - d * BUF), tp1, tp2)
                if r is None: continue
                R, dist, bars = r; busy[mode] = k_now + 1 + bars
                rows.append((tf, pd.Timestamp(now, tz='UTC'), mode, dpos, pat, NAMES[pat][k - 1], d, R, dist))
        print(tf, sum(1 for r in rows if r[0] == tf), flush=True)
    return pd.DataFrame(rows, columns=['tf', 'time', 'stop', 'daily', 'pattern', 'wave', 'side', 'R', 'stop_dist'])


if __name__ == '__main__':
    m1 = load_1m(); T = run(['1h', '15min', '5min'], m1); T.to_csv('ew/daily_map_trades_btc.csv.gz', index=False)
    pd.set_option('display.width', 220); days = (T.time.max() - T.time.min()).days
    g = T.groupby(['stop', 'tf']).agg(n=('R', 'size'), win=('R', lambda x: (x > 0).mean()), avgR=('R', 'mean'), med_stop=('stop_dist', 'median'))
    g['per_day'] = g.n / days; g['R_month'] = g.avgR * g.per_day * 30.4
    T['dwave'] = T.daily.str.split(':').str[-1]
    w = T.groupby(['stop', 'dwave']).R.agg(['size', 'mean']).unstack('stop')
    yr = T[T.stop == 'invalid'].groupby(['tf', T.time.dt.year]).R.mean().unstack('tf')
    txt = ('DAILY MAP -> SUB-WAVE DAY TRADES (BTC, fees included)\n' + g.round(3).to_string() +
           '\n\nBY DAILY WAVE IN PROGRESS\n' + w.round(3).to_string() + '\n\nBY YEAR (stop = invalid)\n' + yr.round(3).to_string())
    print(txt); open('logs/ew_daily_map_report.txt', 'w').write(txt)
