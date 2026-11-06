"""v7: Elliott day-trading on 15m with rule-based trade management.

Setups (big-degree zigzag gives the structure, lower degree gives the entry):
  W3 — wave 2 ends: enter for wave 3. Stop beyond the wave-2 swing (prior low).
       TP1 = 1.0 x wave1 from entry swing, TP2 = 1.618 x wave1.
  W5 — wave 4 ends without overlapping wave 1: enter for wave 5. Stop beyond the
       wave-4 swing, never past the wave-1 extreme (overlap = count dead).
       TP1 = 0.618 x wave1, TP2 = 1.0 x wave1 (equality).
  C  — after A and a B retrace of 0.382-0.886, enter for C. Stop beyond B.
       TP1 = 0.618 x A, TP2 = 1.0 x A (C = A).

Management:
  full   — whole position to TP2 or stop
  half   — half off at TP1, stop on the rest moves to entry (본절), rest to TP2
  trail  — half off at TP1, stop to entry, then trail behind each new
           lower-degree swing until stopped

R: loss of the whole position at the first stop, fees included = -1R.
Selection on 2017-08..2023-10, report 2023-11..2026-10 as unseen data.
"""
import bisect, itertools, os, numpy as np, pandas as pd
FIB = os.environ.get('NOFIB') is None   # NOFIB=1: rules + prior highs/lows only
from engine import atr, zigzag

SPLIT = pd.Timestamp('2023-11-01', tz='UTC')
BUF = 0.0005


def load_15m():
    d = pd.read_csv('data/btcusdt_5m.csv'); d['time'] = pd.to_datetime(d.time, format='ISO8601', utc=True)
    d = d.set_index('time').resample('15min').agg(dict(o='first', h='max', l='min', c='last', v='sum')).dropna()
    return d.reset_index()


df = load_15m(); A = atr(df)
o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
TIME = df.time


def find_setups(K, ratio, maxstop):
    big = zigzag(df, K, A); low = zigzag(df, K / ratio, A)
    lk = [q[3] for q in low]; S = []

    def first_swing(after_known, until_known, kind, test):
        for q in low[bisect.bisect_right(lk, after_known):]:
            if q[3] >= until_known: return None
            if q[2] != kind: continue
            r = test(q)
            if r == 'dead': return None
            if r: return q
        return None

    for j in range(1, len(big)):
        nxt = big[j + 1][3] if j + 1 < len(big) else len(o)
        # --- W3 / C share the shape: leg X (p0->p1) then a counter swing q
        p0, p1 = big[j - 1], big[j]; up = p1[2] == 1; s = 1 if up else -1
        w1 = s * (p1[1] - p0[1])
        if w1 > 0:
            kind = -1 if up else 1
            q = first_swing(p1[3], nxt, kind, lambda q: 'dead' if s * (q[1] - p0[1]) <= 0 else
                            ((0.382 <= s * (p1[1] - q[1]) / w1 < 0.886 if FIB else True) and q[0] > p1[0]))
            if q:
                S.append(('W3', q[3] + 1, up, q[1], q[1] + s * 1.0 * w1 if FIB else p1[1], q[1] + s * 1.618 * w1 if FIB else q[1] + s * w1, None))
                # same shape read as A-B, trading C in the direction of A (opposite to leg X)
            # C: leg X is wave A; q2 = B swing against A
            kindB = 1 if up else -1           # A down when leg X is down: here X up means A up
            # (C setups are generated below from down/up legs symmetrically)
        # --- C: A = p0->p1, B = lower swing retracing A, trade continuation (C)
        # identical geometry to W3 but different targets; tag separately so they can be filtered
        if w1 > 0 and j >= 2:
            prev = big[j - 2]
            # A must follow an opposite leg at least as large (a correction of a prior trend)
            if s * (prev[1] - p0[1]) >= w1:
                q = first_swing(p1[3], nxt, -1 if up else 1, lambda q: 'dead' if s * (q[1] - p0[1]) <= 0 else
                                ((0.382 <= s * (p1[1] - q[1]) / w1 < 0.886 if FIB else True) and q[0] > p1[0]))
                if q:
                    S.append(('C', q[3] + 1, up, q[1], q[1] + s * 0.618 * w1 if FIB else p1[1], q[1] + s * 1.0 * w1, None))
        # --- W5: p0..p3 = waves 1-3 done (0 low,1 high,2 low,3 high for up)
        if j >= 3:
            a0, a1, a2, a3 = big[j - 3], big[j - 2], big[j - 1], big[j]
            up = a3[2] == 1; s = 1 if up else -1
            w1 = s * (a1[1] - a0[1]); w3 = s * (a3[1] - a2[1])
            if w1 > 0 and w3 > 0 and s * (a2[1] - a0[1]) > 0 and w3 >= w1 * 0.618:
                q = first_swing(a3[3], nxt, -1 if up else 1, lambda q: 'dead' if s * (q[1] - a1[1]) <= 0 else
                                ((0.236 <= s * (a3[1] - q[1]) / w3 < 0.618 if FIB else True) and q[0] > a3[0]))
                if q:
                    S.append(('W5', q[3] + 1, up, q[1], q[1] + s * 0.618 * w1 if FIB else a3[1], q[1] + s * 1.0 * w1, a1[1]))
    out = []
    for name, e, up, sw, tp1, tp2, hard in S:
        if e >= len(o): continue
        s = 1 if up else -1; entry = o[e]
        stop = sw * (1 - s * BUF)
        if hard is not None and s * (stop - hard) < 0: stop = hard
        d = s * (entry - stop) / entry
        if d <= 0 or d > maxstop or s * (tp1 - entry) <= 0: continue
        out.append(dict(setup=name, e=e, up=up, entry=entry, stop=stop, tp1=tp1, tp2=tp2, d=d))
    out.sort(key=lambda x: x['e'])
    return out, low


def manage(x, mode, fee, low, lk):
    s = 1 if x['up'] else -1; e = x['e']; entry = x['entry']; stop = x['stop']
    tp1, tp2, d = x['tp1'], x['tp2'], x['d']
    unit = d + 2 * fee                                 # loss per 1.0 position at first stop, incl fees
    pos = 1.0; pnl = 0.0; half_done = False; t = e
    li = bisect.bisect_right(lk, e)
    while t < len(o):
        if mode == 'trail' and half_done:
            while li < len(low) and low[li][3] <= t:
                q = low[li]; li += 1
                if q[2] == (-1 if s > 0 else 1) and s * (q[1] - stop) > 0:
                    stop = q[1] * (1 - s * BUF)
        hit_stop = (l[t] <= stop) if s > 0 else (h[t] >= stop)
        if hit_stop:
            pnl += pos * (s * (stop - entry) / entry - 2 * fee); pos = 0; break
        if mode != 'full' and not half_done and ((h[t] >= tp1) if s > 0 else (l[t] <= tp1)):
            pnl += 0.5 * (s * (tp1 - entry) / entry - 2 * fee); pos = 0.5; half_done = True
            stop = entry * (1 + s * 2 * fee)          # 본절 (fees covered)
            t += 1; continue
        if mode != 'trail' and ((h[t] >= tp2) if s > 0 else (l[t] <= tp2)):
            pnl += pos * (s * (tp2 - entry) / entry - 2 * fee); pos = 0; break
        t += 1
    if pos > 0:
        t = len(o) - 1; pnl += pos * (s * (c[t] - entry) / entry - 2 * fee)
    return pnl / unit, t


def backtest(setups, low, mode, fee, which):
    lk = [q[3] for q in low]; busy = -1; rows = []
    for x in setups:
        if x['setup'] not in which or x['e'] <= busy: continue
        R, t = manage(x, mode, fee, low, lk)
        rows.append((TIME[x['e']], x['setup'], R, x['d'])); busy = t
    return pd.DataFrame(rows, columns=['time', 'setup', 'R', 'd'])


def summ(T, f=0.04):                     # equity risk per trade: 40% margin x 10% margin loss
    if len(T) < 20: return dict(n=len(T))
    days = max((T.time.iloc[-1] - T.time.iloc[0]).days, 1)
    eq = np.cumprod(1 + f * T.R.values)
    mon = (1 + f * T.R).groupby(T.time.dt.strftime('%Y-%m')).prod() - 1
    return dict(n=len(T), per_day=len(T) / days, win=(T.R > 0).mean(), avgR=T.R.mean(),
                R_month=T.R.sum() / (days / 30.4), mdd=(eq / np.maximum.accumulate(eq) - 1).min(),
                med_month=mon.median(), best_month=mon.max(), worst_month=mon.min())


if __name__ == '__main__':
    rows = []
    for K, ra, ms in itertools.product([3, 4, 6], [3, 4], [0.10]):   # stop = prior swing; leverage = 10% / distance
        S, low = find_setups(K, ra, ms)
        for which, mode, fee in itertools.product([('W3',), ('W5',), ('C',), ('W3', 'W5', 'C')],
                                                  ['full', 'half', 'trail'], [0.00035, 0.00025]):
            T = backtest(S, low, mode, fee, which)
            if T.empty: continue
            a = summ(T[T.time < SPLIT]); b = summ(T[T.time >= SPLIT])
            rows.append(dict(K=K, ratio=ra, maxstop=ms, setups='+'.join(which), mode=mode, fee=fee,
                             **{f'{k}_in': v for k, v in a.items()}, **{f'{k}_out': v for k, v in b.items()}))
        print(K, ra, ms, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv('grid7.csv' if FIB else 'grid7_nofib.csv', index=False)
    pd.set_option('display.width', 320)
    ok = R[(R.n_in >= 50) & (R.n_out >= 30)].copy(); ok['m'] = ok[['avgR_in', 'avgR_out']].min(axis=1)
    print('configs', len(R), 'positive both', (ok.m > 0).sum())
    cols = ['K', 'ratio', 'maxstop', 'setups', 'mode', 'fee', 'n_in', 'win_in', 'avgR_in', 'per_day_out', 'win_out',
            'avgR_out', 'R_month_out', 'mdd_out', 'med_month_out', 'worst_month_out']
    print(ok.sort_values('m', ascending=False)[cols].head(20).round(3).to_string(index=False))
