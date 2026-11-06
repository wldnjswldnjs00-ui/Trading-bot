"""Heikin-Ashi + 200 EMA strategy, 1:3, one trade at a time.

Signal (long): HA close > EMA200 (of HA close), HA candle green, HA body bigger
than the previous HA body. Short = mirror. Entry next bar open (real price).
Stop = real low (high for shorts) of the signal bar. Target = 3 x risk.
Money: $300 start, margin 20% of current equity, 10x -> notional 2 x equity.
Fees 0.035% a side. Loss on a trade is capped at the margin (liquidation at
-10% price, ~maintenance ignored). One position at a time across all coins:
after a trade closes, the next signal (any coin) is taken.
"""
import sys, glob, os, zipfile, numpy as np, pandas as pd
from numba import njit

FEE, MARGIN, LEV, START = 0.00035, 0.20, 10.0, 300.0
PERIOD = (pd.Timestamp('2023-11-01', tz='UTC'), pd.Timestamp('2026-11-01', tz='UTC'))


@njit(cache=True)
def heikin(o, h, l, c):
    n = len(c); ho = np.empty(n); hc = (o + h + l + c) / 4
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, n): ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return ho, hc


@njit(cache=True)
def ema(x, n):
    out = np.empty(len(x)); a = 2 / (n + 1); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


@njit(cache=True)
def trades(o, h, l, c, side, rr, maxbars):
    """Every signal's own outcome. Returns entry idx, exit idx, price return (no fees), stop distance."""
    ho, hc = heikin(o, h, l, c); e = ema(hc, 200); n = len(c)
    body = hc - ho
    E = np.empty(n, np.int64); X = np.empty(n, np.int64); R = np.empty(n); Dd = np.empty(n); m = 0
    for i in range(201, n - 1):
        if side > 0:
            sig = hc[i] > e[i] and body[i] > 0 and abs(body[i]) > abs(body[i - 1])
        else:
            sig = hc[i] < e[i] and body[i] < 0 and abs(body[i]) > abs(body[i - 1])
        if not sig: continue
        px = o[i + 1]; stop = l[i] if side > 0 else h[i]
        d = side * (px - stop) / px
        if d <= 0: continue
        tgt = px * (1 + side * rr * d); liq = px * (1 - side * 0.1)
        st = stop if side * (stop - liq) > 0 else liq
        res = 0.0; j = i + 1; done = False
        while j < min(n, i + 1 + maxbars):
            if (l[j] <= st) if side > 0 else (h[j] >= st):
                res = side * (st - px) / px; done = True; break
            if (h[j] >= tgt) if side > 0 else (l[j] <= tgt):
                res = side * (tgt - px) / px; done = True; break
            j += 1
        if not done: j = min(n - 1, j); res = side * (c[j] - px) / px
        E[m] = i + 1; X[m] = j; R[m] = res; Dd[m] = d; m += 1
    return E[:m], X[:m], R[:m], Dd[:m]


def bars(df, tf):
    if tf in ('1min',) or (tf == '5min' and df.index[1] - df.index[0] == pd.Timedelta('5min')): return df
    return df.resample(tf, origin='epoch').agg(dict(o='first', h='max', l='min', c='last')).dropna()


def load_btc_1m():
    D = np.load('data/btc_spot_1m.npy')
    return pd.DataFrame(D[:, 1:5], columns=list('ohlc'), index=pd.to_datetime(D[:, 0].astype(np.int64), unit='ms', utc=True))


def load_coin(sym):
    fr = []
    for f in sorted(glob.glob(f'data/alt5m/{sym}USDT-5m-*.zip')):
        z = zipfile.ZipFile(f); d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]
        fr.append(d.iloc[:, :5].astype(float))
    if not fr: return None
    d = pd.concat(fr); d.columns = ['t', 'o', 'h', 'l', 'c']; d = d[d.t % 300000 == 0].drop_duplicates('t')
    return d.set_index(pd.to_datetime(d.t.astype(np.int64), unit='ms', utc=True))[['o', 'h', 'l', 'c']].sort_index()


def signals(df, tf, side, sym, rr=3.0):
    b = bars(df, tf); b = b[(b.index >= PERIOD[0] - pd.Timedelta('30D')) & (b.index < PERIOD[1])]
    if len(b) < 500: return pd.DataFrame()
    E, X, R, D = trades(b.o.values, b.h.values, b.l.values, b.c.values, side, rr, 20000)
    t = b.index
    T = pd.DataFrame(dict(sym=sym, entry=t[E], exit=t[X], r=R, stop=D, side=side))
    return T[T.entry >= PERIOD[0]]


def sequential(T):
    """One trade at a time: after a trade exits, take the next signal (any coin)."""
    T = T.sort_values('entry').reset_index(drop=True)
    eq = START; free_at = PERIOD[0]; rows = []
    ent, ext, r = T.entry.values, T.exit.values, T.r.values
    i = 0; n = len(T)
    while i < n:
        if ent[i] < np.datetime64(free_at):
            i = np.searchsorted(ent, np.datetime64(free_at), side='left')
            continue
        notional = eq * MARGIN * LEV
        pnl = notional * (r[i] - 2 * FEE)
        pnl = max(pnl, -eq * MARGIN)                       # isolated margin: cannot lose more than the margin
        eq += pnl; rows.append((ent[i], ext[i], r[i], pnl, eq, T.sym.values[i]))
        free_at = pd.Timestamp(ext[i]).tz_localize('UTC') if pd.Timestamp(ext[i]).tzinfo is None else pd.Timestamp(ext[i])
        if eq <= 1: break
        i += 1
    return pd.DataFrame(rows, columns=['entry', 'exit', 'r', 'pnl', 'equity', 'sym'])


def summary(S, label):
    if S.empty: return dict(test=label, trades=0)
    days = (PERIOD[1] - PERIOD[0]).days
    S['month'] = pd.to_datetime(S.entry).dt.strftime('%Y-%m')
    eq_m = S.groupby('month').equity.last(); prev = eq_m.shift(1).fillna(START); mret = eq_m / prev - 1
    peak = S.equity.cummax()
    return dict(test=label, trades=len(S), per_day=round(len(S) / days, 2), win=round((S.pnl > 0).mean(), 3),
                final_usd=round(S.equity.iloc[-1], 2), median_month=f'{mret.median():+.1%}', best_month=f'{mret.max():+.1%}',
                worst_month=f'{mret.min():+.1%}', max_drawdown=f'{(S.equity / peak - 1).min():.1%}')


if __name__ == '__main__':
    btc1 = load_btc_1m(); out = []
    syms = open('universe150.txt').read().split()
    coins = {s: load_coin(s) for s in syms}
    coins = {s: d for s, d in coins.items() if d is not None and len(d) > 20000}
    print('coins loaded', len(coins), flush=True)
    for tf in ('1min', '5min', '15min', '1h'):
        for side, sname in ((1, 'long'), (-1, 'short')):
            Tb = signals(btc1, tf, side, 'BTC')
            out.append(summary(sequential(Tb), f'BTC only | {tf} | {sname}'))
            if tf == '1min': continue
            Ta = pd.concat([signals(d, tf, side, s) for s, d in coins.items()] + [Tb])
            out.append(summary(sequential(Ta), f'150 coins | {tf} | {sname}'))
            print(tf, sname, 'signals', len(Ta), flush=True)
        # long + short together (BTC)
    R = pd.DataFrame(out); pd.set_option('display.width', 250)
    txt = ('HEIKIN-ASHI + EMA200, 1:3, one trade at a time, $300, 20% margin x10, fees 0.035%/side, 2023-11..2026-10\n'
           + R.to_string(index=False))
    print(txt); open('logs/ha_report.txt', 'w').write(txt)
