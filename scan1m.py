"""Scalping rule scan on BTCUSDT perpetual 1m (last 3 years).

Owner's sizing: every trade loses exactly RISK of equity at the stop, fees
included; leverage = RISK / (stop distance + round-trip fee), so a tight stop
means higher leverage. Notional cap = 40% margin x 20x... raised per owner when
the stop is tight, but we report the leverage each trade would need.

Train = first 2 years, test = last year (never used to pick rules).
Entry at next bar open after the signal (market / taker). Stop checked before
target inside the same bar (conservative). One position at a time.
"""
import itertools, numpy as np, pandas as pd
from numba import njit

D = np.load('data/btc_fut_1m.npy')
t, o, h, l, c, v = (D[:, i] for i in range(6))
N = len(c)
SPLIT = np.searchsorted(t, pd.Timestamp('2025-11-01', tz='UTC').value // 10**6)
DAYS_IN, DAYS_OUT = SPLIT / 1440, (N - SPLIT) / 1440


def rolling(x, n, f):
    s = pd.Series(x)
    return getattr(s.rolling(n, min_periods=n), f)().values


def atr(n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return rolling(tr, n, 'mean')


ATR = atr(14)


@njit(cache=True)
def sim(sig, stopd, b, fee, hold, o, h, l, c):
    """sig[i] = +1/-1 at close of bar i; stopd[i] = stop distance (fraction).
    Returns entry index and net R per trade (loss at stop incl fees = -1R)."""
    n = len(c); ent = np.empty(n, np.int64); R = np.empty(n); k = 0; i = 0
    while i < n - 2:
        s = sig[i]
        if s == 0 or not (stopd[i] > 0):
            i += 1; continue
        e = i + 1; px = o[e]; d = stopd[i]
        stop = px * (1 - s * d); tgt = px * (1 + s * b * d)
        out = c[min(n - 1, e + hold)]; j_end = min(n - 1, e + hold)
        j = e
        while j <= j_end:
            if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                out = stop; break
            if (s > 0 and h[j] >= tgt) or (s < 0 and l[j] <= tgt):
                out = tgt; break
            j += 1
        if j > j_end: j = j_end
        pnl = s * (out - px) / px - 2 * fee
        R[k] = pnl / (d + 2 * fee); ent[k] = e; k += 1
        i = j + 1
    return ent[:k], R[:k]


def signals():
    """Yield (family, params, sig array)."""
    for n in (5, 15, 30, 60):                                       # breakout
        hh = rolling(h, n, 'max'); ll = rolling(l, n, 'min')
        hh = np.r_[np.nan, hh[:-1]]; ll = np.r_[np.nan, ll[:-1]]
        yield 'breakout', f'n={n}', np.where(c > hh, 1, np.where(c < ll, -1, 0)).astype(np.int8)
        yield 'fade_break', f'n={n}', np.where(c > hh, -1, np.where(c < ll, 1, 0)).astype(np.int8)
    for n, z in itertools.product((20, 60), (2.0, 3.0)):          # band fade
        m = rolling(c, n, 'mean'); sd = rolling(c, n, 'std')
        yield 'band_fade', f'n={n},z={z}', np.where(c < m - z * sd, 1, np.where(c > m + z * sd, -1, 0)).astype(np.int8)
        yield 'band_break', f'n={n},z={z}', np.where(c > m + z * sd, 1, np.where(c < m - z * sd, -1, 0)).astype(np.int8)
    for f_, s_ in ((9, 50), (20, 200)):                             # trend pullback
        ef = pd.Series(c).ewm(span=f_).mean().values; es = pd.Series(c).ewm(span=s_).mean().values
        pc = np.r_[c[0], c[:-1]]; pef = np.r_[ef[0], ef[:-1]]
        up = (ef > es) & (pc < pef) & (c > ef); dn = (ef < es) & (pc > pef) & (c < ef)
        yield 'pullback', f'ema{f_}/{s_}', np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)
    for n, m_ in itertools.product((60,), (3.0, 5.0)):            # volume spike
        av = np.r_[np.nan, rolling(v, n, 'mean')[:-1]]; spike = v > m_ * av; dirn = np.sign(c - o)
        yield 'vol_momo', f'x{m_}', np.where(spike, dirn, 0).astype(np.int8)
        yield 'vol_fade', f'x{m_}', np.where(spike, -dirn, 0).astype(np.int8)


def stat(R, ent, lo, hi, days):
    m = (ent >= lo) & (ent < hi); r = R[m]
    if len(r) < 50: return dict(n=len(r))
    return dict(n=len(r), per_day=len(r) / days, win=(r > 0).mean(), avgR=r.mean())


if __name__ == '__main__':
    rows = []
    for fam, p, sig in signals():
        for k, b, fee in itertools.product((0.5, 1.0, 2.0), (1.0, 2.0, 3.0, 4.0, 6.0), (0.00035, 0.00025)):
            stopd = k * ATR / c
            ent, R = sim(sig, np.nan_to_num(stopd), b, fee, 120, o, h, l, c)
            a = stat(R, ent, 0, SPLIT, DAYS_IN); z = stat(R, ent, SPLIT, N, DAYS_OUT)
            med_stop = np.nanmedian(stopd[ent - 1]) if len(ent) else np.nan
            rows.append(dict(family=fam, p=p, atr_k=k, tgtR=b, fee=fee, med_stop=med_stop,
                             **{f'{x}_in': y for x, y in a.items()}, **{f'{x}_out': y for x, y in z.items()}))
        print(fam, p, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv('scan1m.csv', index=False)
    pd.set_option('display.width', 300)
    ok = R[(R.n_in >= 50) & (R.n_out >= 50)].copy(); ok['m'] = ok[['avgR_in', 'avgR_out']].min(axis=1)
    print('configs', len(R), '| positive both', (ok.m > 0).sum(), '| best avgR_out', ok.avgR_out.max().round(3))
    print(ok.sort_values('m', ascending=False).head(15)[['family', 'p', 'atr_k', 'tgtR', 'fee', 'med_stop', 'n_in', 'per_day_in',
          'win_in', 'avgR_in', 'per_day_out', 'win_out', 'avgR_out']].round(4).to_string(index=False))
