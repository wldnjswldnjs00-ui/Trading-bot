"""v6: resting limit orders at the wave-2 confluence level (fib x 매물대).

Once wave 1 is known (big-degree zigzag), place a limit order at the fib level
(0.5 / 0.618 / 0.786) with the heaviest volume node nearby. Stop a fixed
distance beyond it. Cancel when price makes a new wave-1 extreme first, or the
next big pivot gets confirmed. All fills/exits simulated on 5m bars.

Fees: entry and take-profit are maker (limit), stop is taker (market).
Sizing per owner: loss at stop incl. fees <= RISK, notional <= 8x equity.
"""
import itertools, numpy as np, pandas as pd
from engine import atr, zigzag
from confluence import vp_strength

SPLIT = pd.Timestamp('2023-01-01', tz='UTC'); MAXN = 8.0
FIBS = (0.5, 0.618, 0.786)


def load(p):
    d = pd.read_csv(p); d['time'] = pd.to_datetime(d.time, format='ISO8601', utc=True); return d


M = load('data/btcusdt_5m.csv')
mt = M.time.values.astype('datetime64[ns]').astype(np.int64)
mo, mh, ml, mc = M.o.values, M.h.values, M.l.values, M.c.values
FRAMES = {'1h': load('data/btcusdt_1h.csv'), '5m': M}


def orders(frame, K, vpwin, need_vp):
    df = FRAMES[frame]; A = atr(df); h, l, v = df.h.values, df.l.values, df.v.values
    t = df.time.values.astype('datetime64[ns]').astype(np.int64)
    big = zigzag(df, K, A); out = []
    for j in range(1, len(big)):
        p0, p1 = big[j - 1], big[j]; up = p1[2] == 1; s = 1 if up else -1
        w1 = s * (p1[1] - p0[1])
        if w1 <= 0: continue
        best = None
        for f in FIBS:
            lvl = p1[1] - s * f * w1
            vp = vp_strength(h, l, v, max(0, p1[0] - vpwin), p1[0], lvl, 0.002)
            if best is None or vp > best[1]: best = (lvl, vp, f)
        if need_vp and best[1] < 1.5: continue
        start = t[p1[3]] + 1                                   # order goes live after wave 1 is known
        end = t[big[j + 1][3]] if j + 1 < len(big) else mt[-1]
        out.append((start, end, up, best[0], p1[1], w1))
    return out


def run(od, stop_pct, tgt, maker, taker, risk):
    tr = []; busy = 0
    for start, end, up, lvl, ext, w1 in od:
        if start < busy: continue
        s = 1 if up else -1
        i = np.searchsorted(mt, start); j = np.searchsorted(mt, end)
        fill = None
        for k in range(i, min(j, len(mo))):
            if (mh[k] >= ext) if up else (ml[k] <= ext): break          # new extreme first: cancel
            if (ml[k] <= lvl) if up else (mh[k] >= lvl):
                fill = (k, min(lvl, mo[k]) if up else max(lvl, mo[k])); break
        if fill is None: continue
        k0, entry = fill
        stop = entry * (1 - s * stop_pct)
        T = entry * (1 + s * tgt * stop_pct) if tgt < 50 else lvl + s * 1.618 * w1
        loss_frac = stop_pct + maker + taker
        notional = min(risk / loss_frac, MAXN)
        px = None
        for k in range(k0, len(mo)):
            if (ml[k] <= stop) if up else (mh[k] >= stop): px, fee = stop, maker + taker; break
            if k > k0 and ((mh[k] >= T) if up else (ml[k] <= T)): px, fee = T, 2 * maker; break
        if px is None: px, fee = mc[-1], maker + taker
        tr.append((M.time[k0], notional * (s * (px - entry) / entry - fee), notional * loss_frac))
        busy = mt[k]
    return pd.DataFrame(tr, columns=['time', 'pnl', 'risk'])


def summ(T, risk):
    if len(T) < 10: return dict(n=len(T))
    R = T.pnl / T.risk
    eq = np.maximum(np.cumprod(1 + T.pnl.values), 0)
    days = max((T.time.iloc[-1] - T.time.iloc[0]).days, 1)
    m = (1 + T.pnl).groupby(T.time.dt.strftime('%Y-%m')).prod() - 1
    return dict(n=len(T), per_day=len(T) / days, win=(R > 0).mean(), avgR=R.mean(),
                mdd=(eq / np.maximum.accumulate(eq) - 1).min(), med_month=m.median(), worst_month=m.min())


if __name__ == '__main__':
    rows = []
    for (frame, K) in [('1h', 4), ('1h', 6), ('5m', 6), ('5m', 8), ('5m', 12)]:
        for need_vp in (False, True):
            od = orders(frame, K, 720 if frame == '1h' else 2016, need_vp)
            for sp, tg, (mk, tk) in itertools.product([0.002, 0.003, 0.005, 0.008], [2, 3, 5, 99],
                                                     [(0.0001, 0.00025), (0.0002, 0.0005)]):
                T = run(od, sp, tg, mk, tk, 0.01)
                a = summ(T[T.time < SPLIT], 0.01); b = summ(T[T.time >= SPLIT], 0.01)
                rows.append(dict(frame=frame, K=K, vp=need_vp, stop=sp, tgt=tg, maker=mk,
                                 **{f'{x}_in': y for x, y in a.items()}, **{f'{x}_out': y for x, y in b.items()}))
            print(frame, K, need_vp, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv('grid6.csv', index=False)
    pd.set_option('display.width', 300)
    ok = R[(R.n_in >= 30) & (R.n_out >= 30)].copy(); ok['m'] = ok[['avgR_in', 'avgR_out']].min(axis=1)
    print('positive both:', (ok.m > 0).sum(), 'of', len(ok))
    print(ok.sort_values('m', ascending=False).head(15)[['frame', 'K', 'vp', 'stop', 'tgt', 'maker', 'n_in', 'avgR_in', 'win_in',
          'n_out', 'per_day_out', 'win_out', 'avgR_out', 'mdd_out', 'med_month_out']].round(3).to_string(index=False))
