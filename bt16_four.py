"""Four additions on top of the current best (ICT 30m ladder + impulse-delta filter,
half at prior high + stop to entry + trail the rest on KT=3 swings):

  sweep    — liquidity sweep entry: a bar wicks beyond the last confirmed swing
             low and closes back above it (in a 30m bull structure); market entry
             next open, stop beyond the wick, first target the last swing high.
  session  — enter only in the session-open windows (UTC 00-02 Asia/KST 9-11,
             07-09 London, 13-16 New York).
  regime   — enter only with the 4h swing structure (HH+HL long / LH+LL short).
  pyramid  — no half exit; each time a new trailing swing confirms, move the
             stop there and add 0.5 x the first size at the next open, at most
             3 adds, only if the whole position still exits at break-even or
             better at the new stop. Worst case stays the first 1R.
One position per coin per variant. R = loss of the first full size at the
first stop, fees included.
"""
import glob, os, bisect, sys, itertools, numpy as np, pandas as pd
from engine import atr, zigzag
from bt13_ict_cvd import load30, MAKER, TAKER, THRU, BUF, SPLIT

LEVELS = (0.5, 0.618, 0.705); KT = 3.0
SESSION_H = {0, 1, 7, 8, 13, 14, 15}


def struct_bias(piv, n):
    """+1/-1/0 per bar from the last four confirmed swings (known time)."""
    ev_i, ev_b = [], []
    for i in range(3, len(piv)):
        q = piv[i - 3:i + 1]
        hs = [p[1] for p in q if p[2] == 1]; ls = [p[1] for p in q if p[2] == -1]
        b = 1 if hs[1] > hs[0] and ls[1] > ls[0] else -1 if hs[1] < hs[0] and ls[1] < ls[0] else 0
        ev_i.append(piv[i][3]); ev_b.append(b)
    out = np.zeros(n, np.int8); j = 0; cur = 0
    for t in range(n):
        while j < len(ev_i) and ev_i[j] <= t: cur = ev_b[j]; j += 1
        out[t] = cur
    return out


def prep(df):
    A = atr(df); n = len(df)
    P = dict(df=df, o=df.o.values, h=df.h.values, l=df.l.values, c=df.c.values, n=n,
             piv=zigzag(df, 3, A), tz=zigzag(df, KT, A), hour=df.time.dt.hour.values)
    P['tk'] = [q[3] for q in P['tz']]
    v = df.v.values; dl = 2 * df.tb.values - v
    P['cd'], P['cv'] = np.r_[0, np.cumsum(dl)], np.r_[0, np.cumsum(v)]
    P['bias30'] = struct_bias(P['piv'], n)
    H4 = df.set_index('time').resample('4h', origin='epoch').agg(dict(o='first', h='max', l='min', c='last', v='sum')).dropna().reset_index()
    pz = zigzag(H4, 3, atr(H4, 50)); b4 = struct_bias(pz, len(H4))
    close4 = (H4.time + pd.Timedelta('4h')).values                   # usable after the 4h bar closes
    idx = np.searchsorted(close4, df.time.values, side='right') - 1
    P['bias4h'] = np.where(idx >= 0, b4[np.clip(idx, 0, None)], 0)
    return P


def share(P, a, b): return (P['cd'][b] - P['cd'][a]) / max(P['cv'][b] - P['cv'][a], 1e-12)


def ict_setups(P):
    piv, o = P['piv'], P['o']; S = []
    for i in range(3, len(piv)):
        p1, p2, p3, p4 = piv[i - 3:i + 1]; up = p4[2] == 1; s = 1 if up else -1
        if not (s * (p3[1] - p1[1]) > 0 and s * (p4[1] - p2[1]) > 0): continue
        leg = s * (p4[1] - p3[1]); start = p4[3] + 1
        if leg <= 0 or start >= P['n'] or s * share(P, p3[0], p4[0] + 1) <= 0: continue
        lv = [p4[1] - s * f * leg for f in LEVELS]
        if (o[start] <= lv[0]) if up else (o[start] >= lv[0]): continue
        S.append(dict(kind='ict', start=start, up=up, lv=lv, stop=p3[1] * (1 - s * BUF), tp1=p4[1], ext=p4[1],
                      inv=p4[1] - s * 0.79 * leg))
    return S


def sweep_setups(P):
    piv, h, l, c, n = P['piv'], P['h'], P['l'], P['c'], P['n']; S = []; used = set()
    known = [p[3] for p in piv]
    for t in range(1, n - 1):
        j = bisect.bisect_right(known, t - 1)                       # swings known before bar t
        if j < 2: continue
        last_low = next((p for p in reversed(piv[max(0, j - 4):j]) if p[2] == -1), None)
        last_high = next((p for p in reversed(piv[max(0, j - 4):j]) if p[2] == 1), None)
        b = P['bias30'][t - 1]
        for up, sw, tgt in ((True, last_low, last_high), (False, last_high, last_low)):
            if sw is None or tgt is None or (sw[0], up) in used: continue
            if b != (1 if up else -1): continue
            s = 1 if up else -1
            if (l[t] < sw[1] and c[t] > sw[1]) if up else (h[t] > sw[1] and c[t] < sw[1]):
                used.add((sw[0], up))
                stop = (l[t] if up else h[t]) * (1 - s * BUF)
                S.append(dict(kind='sweep', start=t + 1, up=up, stop=stop, tp1=tgt[1]))
    return S


def simulate(P, x, pyramid, sess, regime):
    o, h, l, c, n = P['o'], P['h'], P['l'], P['c'], P['n']
    up = x['up']; s = 1 if up else -1; stop = x['stop']; tz, tk = P['tz'], P['tk']
    tr = []                                                         # (size, price, fee_in)
    if x['kind'] == 'sweep':
        e = x['start']
        if e >= n: return None
        px = o[e]; d = s * (px - stop) / px
        if d <= 0 or d > 0.10 or s * (x['tp1'] - px) <= 0: return None
        unit = d + 2 * TAKER; tr.append([1.0, px, TAKER]); ft = e; t = e
    else:
        lv = x['lv']; dfull = [s * (q - stop) / q for q in lv]
        if min(dfull) <= 0 or max(dfull) > 0.10: return None
        w = 1 / 3; unit = sum(w * (d + MAKER + TAKER) for d in dfull)
        filled = [False] * 3; ft = None; t = x['start']
        while t < n:                                                # wait for the first fill
            if ((h[t] > x['ext']) if up else (l[t] < x['ext'])) or ((c[t] < x['inv']) if up else (c[t] > x['inv'])): return None
            hit = [k for k in range(3) if not filled[k] and ((l[t] < lv[k] * (1 - THRU)) if up else (h[t] > lv[k] * (1 + THRU)))]
            if hit:
                for k in hit: filled[k] = True; tr.append([w, lv[k], MAKER])
                ft = t; break
            t += 1
        if ft is None: return None
    if sess and P['hour'][ft] not in SESSION_H: return None
    if regime and P['bias4h'][ft] != s: return None
    half = False; adds = 0; ti = bisect.bisect_right(tk, ft); pnl = 0.0
    size = lambda: sum(q[0] for q in tr)
    avg = lambda: sum(q[0] * q[1] for q in tr) / size()
    trailing_on = pyramid
    while t < n:
        if x['kind'] == 'ict' and not half and not pyramid:          # remaining rungs keep filling until TP1
            for k in range(3):
                if not filled[k] and ((l[t] < lv[k] * (1 - THRU)) if up else (h[t] > lv[k] * (1 + THRU))):
                    filled[k] = True; tr.append([w, lv[k], MAKER])
        if (l[t] <= stop) if up else (h[t] >= stop):
            pnl += sum(q[0] * (s * (stop - q[1]) / q[1] - q[2] - TAKER) for q in tr); tr = []; break
        if not pyramid and not half and t > ft and ((h[t] > x['tp1']) if up else (l[t] < x['tp1'])):
            for q in tr:
                pnl += q[0] / 2 * (s * (x['tp1'] - q[1]) / q[1] - q[2] - MAKER); q[0] /= 2
            half = True; trailing_on = True
            be = avg() * (1 + s * (MAKER + TAKER)); stop = max(stop, be) if up else min(stop, be)
            if x['kind'] == 'ict': filled = [True] * 3
            t += 1; continue
        if trailing_on:
            while ti < len(tz) and tz[ti][3] <= t:
                q = tz[ti]; ti += 1
                if q[2] != (-1 if up else 1) or q[0] < ft: continue
                ns = q[1] * (1 - s * BUF)
                if s * (ns - stop) <= 0: continue
                stop = ns
                if pyramid and adds < 3 and t + 1 < n:
                    px = o[t + 1]; tr.append([0.5, px, TAKER])
                    if s * (stop - avg()) < 0: tr.pop()               # would risk more than break-even: skip
                    else: adds += 1
        t += 1
    if tr:
        t = n - 1; pnl += sum(q[0] * (s * (c[t] - q[1]) / q[1] - q[2] - TAKER) for q in tr)
    return ft, t, pnl / unit, adds


VARIANTS = {f'{e}|{"sess" if se else "-"}|{"reg" if rg else "-"}|{"pyr" if py else "half_trail"}': (e, se, rg, py)
            for e, se, rg, py in itertools.product(('ict', 'sweep'), (False, True), (False, True), (False, True))}

if __name__ == '__main__':
    syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
    if len(sys.argv) > 1: syms = [s for s in syms if s in set(open(sys.argv[1]).read().split())]
    rows = []
    for sym in syms:
        try: df = load30(sym)
        except Exception: continue
        if len(df) < 2000: continue
        P = prep(df); setups = {'ict': ict_setups(P), 'sweep': sweep_setups(P)}
        for name, (e, se, rg, py) in VARIANTS.items():
            busy = -1
            for x in setups[e]:
                if x['start'] <= busy: continue
                r = simulate(P, x, py, se, rg)
                if r is None: continue
                ft, tend, R, adds = r; busy = tend
                rows.append((name, sym, df.time[ft], R, adds))
        print(sym, flush=True)
    T = pd.DataFrame(rows, columns=['cfg', 'sym', 'time', 'R', 'adds']); T['test'] = T.time >= SPLIT
    T.to_csv(f'four_trades_{len(syms)}.csv.gz', index=False)
    days = (T.time.max() - T.time.min()).days; G = T.groupby('cfg')
    win = T[T.R > 0].groupby('cfg').R.mean(); loss = -T[T.R <= 0].groupby('cfg').R.mean()
    S = pd.DataFrame(dict(n=G.size(), per_day=G.size() / days, win=G.R.apply(lambda x: (x > 0).mean()), avg_win=win,
                          payoff=win / loss, avgR=G.R.mean(), first=T[~T.test].groupby('cfg').R.mean(),
                          last=T[T.test].groupby('cfg').R.mean(), coins_pos=T.groupby(['cfg', 'sym']).R.mean().gt(0).groupby('cfg').mean(),
                          R_month=G.R.sum() / (days / 30.4), max_R=G.R.max(), adds=G.adds.mean()))
    S.to_csv(f'four_{len(syms)}.csv'); pd.set_option('display.width', 260)
    print('coins', len(syms)); print(S.sort_values('avgR', ascending=False).round(3).to_string())
