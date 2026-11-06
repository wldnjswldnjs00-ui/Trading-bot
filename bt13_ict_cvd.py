"""ICT 30m (K3, 0.5, half) + order-flow confirmation, 30 coins, 2023-11..2026-10.

delta = taker buy - taker sell (from klines). For a bull setup (mirror for bear):
  A  leg:    net delta share during the impulse (swing 3 -> high) > thr      (limit)
  C  retr:   net delta share during the pullback, up to the bar BEFORE fill > thr (limit)
  B  absorb: the bar that reaches the 0.5 level closes with delta share > thr;
             enter at market on the next open                                 (market)
Stop = prior swing (3), half off at prior high + stop to entry, rest to -0.62 ext.
Fills: limit needs a trade-through; maker on entry/TP, taker on stop.
"""
import glob, os, zipfile, itertools, numpy as np, pandas as pd
from engine import atr, zigzag

MAKER, TAKER, THRU, BUF = 0.00014, 0.00035, 0.0002, 0.0005
SPLIT = pd.Timestamp('2025-11-01', tz='UTC')


def load30(sym):
    fr = []
    for f in sorted(glob.glob(f'data/alt5m/{sym}USDT-5m-*.zip')):
        z = zipfile.ZipFile(f); d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit(): d = d.iloc[1:]
        fr.append(d.iloc[:, [0, 1, 2, 3, 4, 5, 9]].astype(float))
    d = pd.concat(fr); d.columns = ['t', 'o', 'h', 'l', 'c', 'v', 'tb']; d = d[d.t % 300000 == 0].drop_duplicates('t')
    d['time'] = pd.to_datetime(d.t, unit='ms', utc=True)
    d = d.set_index('time').sort_index().resample('30min', origin='epoch').agg(
        dict(o='first', h='max', l='min', c='last', v='sum', tb='sum')).dropna()
    return d.reset_index()


def candidates(df, K=3, level=0.5):
    A = atr(df); piv = zigzag(df, K, A)
    o, h, l, c, v = (df[k].values for k in 'ohlcv'); dl = 2 * df.tb.values - v
    cd, cv = np.r_[0, np.cumsum(dl)], np.r_[0, np.cumsum(v)]
    share = lambda a, b: (cd[b] - cd[a]) / max(cv[b] - cv[a], 1e-12)      # bars a..b-1
    out = []
    for i in range(3, len(piv)):
        p1, p2, p3, p4 = piv[i - 3], piv[i - 2], piv[i - 1], piv[i]
        up = p4[2] == 1; s = 1 if up else -1
        if not (s * (p3[1] - p1[1]) > 0 and s * (p4[1] - p2[1]) > 0): continue
        leg = s * (p4[1] - p3[1])
        if leg <= 0: continue
        lvl = p4[1] - s * level * leg; inv = p4[1] - s * 0.79 * leg
        start = p4[3] + 1; end = piv[i + 1][3] if i + 1 < len(piv) else len(o)
        if start >= len(o) or ((o[start] <= lvl) if up else (o[start] >= lvl)): continue
        legd = s * share(p3[0], p4[0] + 1)
        for t in range(start, min(end + 1, len(o))):
            if (h[t] > p4[1]) if up else (l[t] < p4[1]): break
            touched = (l[t] <= lvl) if up else (h[t] >= lvl)
            through = (l[t] < lvl * (1 - THRU)) if up else (h[t] > lvl * (1 + THRU))
            if touched:
                out.append(dict(t=t, up=up, lvl=lvl, through=through, stop=p3[1] * (1 - s * BUF), tp1=p4[1],
                                tp2=p4[1] + s * 0.62 * leg, legd=legd, retrd=s * share(p4[0] + 1, t),
                                bard=s * dl[t] / max(v[t], 1e-12)))
                break
            if (c[t] < inv) if up else (c[t] > inv): break
    return out


def trade(df, x, entry, e, entry_fee):
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    s = 1 if x['up'] else -1; stop = x['stop']; d = s * (entry - stop) / entry
    if d <= 0 or d > 0.10: return None
    unit = d + entry_fee + TAKER; pos = 1.0; pnl = 0.0; half = False; t = e
    while t < len(o):
        if (l[t] <= stop) if s > 0 else (h[t] >= stop):
            pnl += pos * (s * (stop - entry) / entry - entry_fee - TAKER); pos = 0; break
        if t > e and not half and ((h[t] > x['tp1']) if s > 0 else (l[t] < x['tp1'])):
            pnl += 0.5 * (s * (x['tp1'] - entry) / entry - entry_fee - MAKER); pos = 0.5; half = True
            stop = entry * (1 + s * (entry_fee + TAKER)); t += 1; continue
        if t > e and ((h[t] > x['tp2']) if s > 0 else (l[t] < x['tp2'])):
            pnl += pos * (s * (x['tp2'] - entry) / entry - entry_fee - MAKER); pos = 0; break
        t += 1
    if pos > 0: t = len(o) - 1; pnl += pos * (s * (c[t] - entry) / entry - entry_fee - TAKER)
    return pnl / unit, t


RULES = [('none', None, None)] + [('A_leg', 'legd', th) for th in (0.0, 0.05, 0.1)] + \
        [('C_retr', 'retrd', th) for th in (-0.05, 0.0, 0.05)] + [('B_absorb', 'bard', th) for th in (0.0, 0.1, 0.2)] + \
        [('A+C', ('legd', 'retrd'), (0.0, 0.0)), ('A+C', ('legd', 'retrd'), (0.05, -0.05))]

if __name__ == '__main__':
    syms = sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
    rows = []
    for sym in syms:
        df = load30(sym); C = candidates(df); o = df.o.values
        for name, key, th in RULES:
            busy = -1
            for x in C:
                if name == 'B_absorb':
                    if x[key] <= th: continue
                    e = x['t'] + 1
                    if e >= len(o) or e <= busy: continue
                    r = trade(df, x, o[e], e, TAKER)
                else:
                    if name == 'A+C':
                        if not (x[key[0]] > th[0] and x[key[1]] > th[1]): continue
                    elif key and x[key] <= th: continue
                    if not x['through'] or x['t'] <= busy: continue
                    r = trade(df, x, x['lvl'], x['t'], MAKER)
                if r is None: continue
                R, busy = r
                rows.append((f'{name}|{th}', sym, df.time[x['t']], R))
        print(sym, flush=True)
    T = pd.DataFrame(rows, columns=['rule', 'sym', 'time', 'R']); T['test'] = T.time >= SPLIT
    days = (T.time.max() - T.time.min()).days
    G = T.groupby('rule')
    S = pd.DataFrame(dict(n=G.size(), per_day=G.size() / days, win=G.R.apply(lambda x: (x > 0).mean()), avgR=G.R.mean(),
                          first=T[~T.test].groupby('rule').R.mean(), last=T[T.test].groupby('rule').R.mean(),
                          coins_pos=T.groupby(['rule', 'sym']).R.mean().gt(0).groupby('rule').mean()))

    def grow(x, f):
        x = x.sort_values('time'); eq = np.cumprod(1 + f * x.R.values)
        return pd.Series({f'x_{f}': eq[-1], f'mdd_{f}': (eq / np.maximum.accumulate(eq) - 1).min()})
    for f in (0.01, 0.02):
        S = S.join(T.groupby('rule').apply(lambda x: grow(x, f)))
    S.to_csv('ict_cvd.csv'); pd.set_option('display.width', 250)
    print(S.sort_values('avgR', ascending=False).round(3).to_string())
