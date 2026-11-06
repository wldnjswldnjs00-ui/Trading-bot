"""Daily map + the whole Elliott catalogue on the small degrees, BTC and 150 coins.

MAP     BTC daily best count -> wave in progress and its direction.
ALLOWED small patterns per daily wave (Frost & Prechter):
        1, A      -> IMP, LD            3 -> IMP            5, C -> IMP, ED
        2         -> ZZ FL DZ DT TZ TT  (no triangle)
        4, B      -> ZZ FL DZ DT TZ TT TRI
        X         -> any corrective     triangle legs -> ZZ FL DZ (E may be TRI)
SETUPS  every position of every small pattern (setups_all):
        trend legs inside the pattern in the daily direction (IMP 2->3, 4->5;
        LD/ED 2->3, 4->5; ZZ/FL B->C; DZ/DT/TZ/TT X->Y/Z; TRI leg->next leg)
        and pattern-complete reversals (small pattern done = daily wave done ->
        trade the next daily wave: IMP/truncated -> back to 4, LD -> deep pullback,
        ED -> back to its start, ZZ/FL/doubles -> resume, TRI -> thrust).
REGIME  optional overlay standing in for the owner's bull/bear/range button:
        hindsight month direction (perfect caller) and the same shifted 7 / 14
        days late (a late caller), to bound what the button can add.
Fees 0.035% a side, R includes fees, one position per coin and degree.
"""
import sys, os, glob, zipfile, numpy as np, pandas as pd
from multiprocessing import Pool
from ew.v4 import Deg, PATTERNS, NAMES, check
from ew.nested import best_counts
from ew.v4_run import parent_lookup
from ew.engine import load_1m

FEE, BUF = 0.00035, 0.0005
ALLOWED = {'1': {'IMP', 'LD'}, 'A': {'IMP', 'LD'}, '3': {'IMP'}, '5': {'IMP', 'ED'}, 'C': {'IMP', 'ED'},
           '2': {'ZZ', 'FL', 'DZ', 'DT', 'TZ', 'TT'}, '4': {'ZZ', 'FL', 'DZ', 'DT', 'TZ', 'TT', 'TRI'},
           'B': {'ZZ', 'FL', 'DZ', 'DT', 'TZ', 'TT', 'TRI'}, 'X': {'ZZ', 'FL', 'DZ', 'DT', 'TZ', 'TT', 'TRI'},
           'D': {'ZZ', 'FL', 'DZ'}, 'E': {'ZZ', 'FL', 'DZ', 'TRI'}, 'W': {'ZZ', 'FL'}, 'Y': {'ZZ', 'FL', 'TRI'}, 'Z': {'ZZ', 'FL', 'TRI'}}


def setups_all(pat, k, P, s):
    """Every tradeable next move for a count at position k.
    -> list of (kind, dir, invalid, swing, tp1, tp2)."""
    n = PATTERNS[pat]; out = []; A = abs(P[1] - P[0])
    if k == n:                                                    # pattern complete -> reversal / resumption
        last = P[-1]
        if pat == 'IMP': out.append(('done', -s, last, last, P[4], last - s * 0.618 * abs(last - P[0])))
        elif pat == 'LD': out.append(('done', -s, last, last, last - s * 0.5 * abs(last - P[0]), last - s * 0.786 * abs(last - P[0])))
        elif pat == 'ED': out.append(('done', -s, last, last, P[4], P[0]))
        elif pat == 'TRI': out.append(('done', -s, last, last, P[4], last - s * A))
        else: out.append(('done', -s, last, last, P[-2], P[0]))
        return out
    nxt = s if k % 2 == 0 else -s                                 # legs alternate
    if pat in ('IMP', 'LD', 'ED'):
        if k == 2: out.append(('trend', s, P[0], P[2], P[1], P[2] + s * (1.618 if pat == 'IMP' else 1.0) * A))
        if k == 4: out.append(('trend', s, P[1] if pat == 'IMP' else P[0], P[4], P[3], P[4] + s * (A if pat == 'IMP' else 0.618 * A)))
    elif pat in ('ZZ',) and k == 2: out.append(('trend', s, P[0], P[2], P[1], P[2] + s * A))
    elif pat == 'FL' and k == 2:
        B = abs(P[2] - P[1]); out.append(('trend', s, P[1] - s * 1.618 * A, P[2], P[1], P[2] + s * (1.618 * A if B > 1.05 * A else 1.0 * A)))
    elif pat in ('DZ', 'DT', 'TZ', 'TT') and k % 2 == 0: out.append(('trend', s, P[0], P[k], P[k - 1], P[k] + s * A))
    elif pat == 'TRI' and k >= 1:
        prev = abs(P[k] - P[k - 1]); out.append(('leg', nxt, P[k], P[k], P[k] + nxt * 0.5 * prev, P[k] + nxt * 0.8 * prev))
    return out


def trade(h, l, k0, px, d, stop, tp1, tp2, maxbars=6000):
    dist = d * (px - stop) / px
    if dist <= 0 or d * (tp1 - px) <= 0 or d * (tp2 - tp1) <= 0: return None
    unit = dist + 2 * FEE; half = False; st = stop
    for t in range(k0, min(len(h), k0 + maxbars)):
        if (l[t] <= st) if d > 0 else (h[t] >= st):
            leg = d * (st - px) / px - 2 * FEE
            return ((0.5 * (d * (tp1 - px) / px - 2 * FEE) + 0.5 * leg) if half else leg) / unit, dist, t - k0
        if not half and ((h[t] >= tp1) if d > 0 else (l[t] <= tp1)):
            half = True; st = px * (1 + d * 2 * FEE); continue
        if half and ((h[t] >= tp2) if d > 0 else (l[t] <= tp2)):
            return (0.5 * (d * (tp1 - px) / px - 2 * FEE) + 0.5 * (d * (tp2 - px) / px - 2 * FEE)) / unit, dist, t - k0
    return None


def run_degree(D, dtl, tf, sym):
    rows = []; busy = {'invalid': -1, 'swing': -1}
    for j in range(1, len(D.p)):
        now = D.kn[j]; dpos, ddir = parent_lookup(dtl, now)
        if dpos == 'none' or ddir == 0: continue
        dwave = dpos.split(':')[-1]
        if dwave not in ALLOWED: continue
        allow = ALLOWED[dwave]
        k_now = np.searchsorted(D.ct, now, side='left')
        if k_now + 1 >= len(D.c): break
        px = D.c[k_now]; cands = []
        for pat, n in PATTERNS.items():
            if pat not in allow: continue
            for k in range(1, n + 1):
                st = j - k
                if st < 0: continue
                P = list(D.p[st:j + 1]); s = 1 if P[1] > P[0] else -1
                if s != ddir: continue                                 # the small pattern must be building the daily wave
                S = [D.structure(st + q, st + q + 1) for q in range(k)]
                ok, sc, _, info = check(pat, P, s, S, D, D.i[st:j + 1], dpos)
                if not ok: continue
                for su in setups_all(pat, k, P, s):
                    kind, d = su[0], su[1]
                    if kind in ('trend', 'leg') and d != ddir and pat != 'TRI': continue
                    cands.append((sc + 0.04 * k, pat, k, su, info.get('trunc', False)))
        if not cands: continue
        sc, pat, k, (kind, d, inv, sw, tp1, tp2), trunc = max(cands, key=lambda x: x[0])
        for mode, lvl in (('invalid', inv), ('swing', sw)):
            if k_now + 1 <= busy[mode]: continue
            r = trade(D.h, D.l, k_now + 1, px, d, lvl * (1 - d * BUF), tp1, tp2)
            if r is None: continue
            R, dist, bars = r; busy[mode] = k_now + 1 + bars
            name = f'{pat}{"_trunc" if trunc else ""}'
            rows.append((sym, tf, pd.Timestamp(now, tz='UTC'), mode, dwave, name, NAMES[pat][k - 1], kind, d, R, dist))
    return rows


def load_coin(sym):
    fr = []
    for f in sorted(glob.glob(f'data/alt5m/{sym}USDT-5m-*.zip')):
        z = zipfile.ZipFile(f); d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]
        fr.append(d.iloc[:, [0, 1, 2, 3, 4, 5, 9]].astype(float))
    d = pd.concat(fr); d.columns = ['t', 'o', 'h', 'l', 'c', 'v', 'tb']; d = d[d.t % 300000 == 0].drop_duplicates('t')
    df = d.set_index(pd.to_datetime(d.t.astype(np.int64), unit='ms', utc=True))[['o', 'h', 'l', 'c', 'v', 'tb']].sort_index()
    return df


_DTL = None


def _init(dtl):
    global _DTL; _DTL = dtl


def work(args):
    sym, tfs = args
    try:
        m = load_coin(sym)
        if len(m) < 50000: return []
        rows = []
        for tf in tfs: rows += run_degree(Deg(tf, m), _DTL, tf, sym)
        return rows
    except Exception as e:
        print('skip', sym, e, flush=True); return []


COLS = ['sym', 'tf', 'time', 'stop', 'dwave', 'pattern', 'wave', 'kind', 'side', 'R', 'stop_dist']

if __name__ == '__main__':
    m1 = load_1m(); dtl = best_counts(Deg('1D', m1), None)[1]
    rows = []
    for tf in ('1h', '15min', '5min'): rows += run_degree(Deg(tf, m1), dtl, tf, 'BTC9y')
    print('BTC 9y rows', len(rows), flush=True)
    syms = [s for s in open('universe150.txt').read().split()]
    with Pool(4, initializer=_init, initargs=(dtl,)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, [(s, ('1h', '15min')) for s in syms])):
            rows += r
            if i % 10 == 0: print('coins done', i + 1, 'rows', len(rows), flush=True)
    T = pd.DataFrame(rows, columns=COLS); T.to_csv('ew/full_map_trades.csv.gz', index=False)
    print('saved', len(T), flush=True)
