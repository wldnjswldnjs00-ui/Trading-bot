"""v8: ICT premium/discount + OTE entries, HTF bias -> LTF entry.

Bull (mirror for bear), on the entry timeframe (LTF):
  1 low, 2 high, 3 higher low, then a higher high (break of structure).
  Leg = 3 -> new high. Once that high is confirmed (price turned), wait for
  "4": price trades back to the entry level (0.5 / 0.618 / 0.705 of the leg).
  Enter at market on the touch. Cancel if a candle body closes beyond 0.79
  first, or price makes a new high first.
  Stop: just beyond swing 3 (prior low). TP1: the leg high (prior high).
  Modes: full  -> all out at prior high
         half  -> half at prior high, rest stop to entry (본절), rest to -0.62 ext
HTF bias (optional): the 12h structure (candles open 00:00/12:00 UTC = 9am/9pm KST)
  or 4h for 15m; trade only with it.
Sizing per owner: margin loss at stop incl fees = 10% -> equity risk 4%.
Pick on 2017-08..2023-10, report 2023-11..2026-10 unseen.
"""
import itertools, numpy as np, pandas as pd
from engine import atr, zigzag

SPLIT = pd.Timestamp('2023-11-01', tz='UTC'); BUF = 0.0005; RISK = 0.04
B5 = pd.read_csv('data/btcusdt_5m.csv'); B5['time'] = pd.to_datetime(B5.time, format='ISO8601', utc=True)
B5 = B5.set_index('time')


def bars(rule):
    d = B5.resample(rule, origin='epoch').agg(dict(o='first', h='max', l='min', c='last', v='sum')).dropna()
    return d.reset_index()


def htf_bias(rule, K, ltf_time):
    """+1 / -1 / 0 per LTF bar from the HTF zigzag structure known at that time."""
    H = bars(rule); A = atr(H, 50); piv = zigzag(H, K, A)
    close_t = (H.time + pd.Timedelta(rule)).values          # HTF info usable after its bar closes
    ev_t, ev_b = [], []
    for i in range(3, len(piv)):
        a, b, c_, d = piv[i - 3], piv[i - 2], piv[i - 1], piv[i]
        highs = [p[1] for p in (a, b, c_, d) if p[2] == 1]; lows = [p[1] for p in (a, b, c_, d) if p[2] == -1]
        bias = 1 if highs[1] > highs[0] and lows[1] > lows[0] else -1 if highs[1] < highs[0] and lows[1] < lows[0] else 0
        ev_t.append(close_t[d[3]]); ev_b.append(bias)
    idx = np.searchsorted(np.array(ev_t, dtype='datetime64[ns]'), ltf_time.values, side='right') - 1
    out = np.where(idx >= 0, np.array(ev_b)[np.clip(idx, 0, None)], 0)
    return out


def setups(df, K, level):
    A = atr(df); piv = zigzag(df, K, A)
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    S = []
    for i in range(3, len(piv)):
        p1, p2, p3, p4 = piv[i - 3], piv[i - 2], piv[i - 1], piv[i]
        up = p4[2] == 1; s = 1 if up else -1
        if not (s * (p3[1] - p1[1]) > 0 and s * (p4[1] - p2[1]) > 0): continue    # HL + BOS (or LH + BOS)
        leg = s * (p4[1] - p3[1])
        if leg <= 0: continue
        lvl = p4[1] - s * level * leg; inv = p4[1] - s * 0.79 * leg
        start = p4[3] + 1; end = piv[i + 1][3] if i + 1 < len(piv) else len(o)
        for t in range(start, min(end + 1, len(o))):
            if (h[t] > p4[1]) if up else (l[t] < p4[1]): break                     # new extreme first
            if (l[t] <= lvl) if up else (h[t] >= lvl):
                entry = min(lvl, o[t]) if up else max(lvl, o[t])
                if s * (entry - inv) < 0: break                                     # gapped past 0.79
                S.append(dict(e=t, up=up, entry=entry, stop=p3[1] * (1 - s * BUF), tp1=p4[1],
                              tp2=p4[1] + s * 0.62 * leg))
                break
            if (c[t] < inv) if up else (c[t] > inv): break                          # body closed past 0.79
    return S


def run(df, S, bias, mode, fee):
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    rows = []; busy = -1
    for x in S:
        e = x['e']; s = 1 if x['up'] else -1
        if e <= busy: continue
        if bias is not None and bias[e] != s: continue
        entry, stop = x['entry'], x['stop']; d = s * (entry - stop) / entry
        if d <= 0 or d > 0.10: continue
        unit = d + 2 * fee; pos = 1.0; pnl = 0.0; half = False; t = e
        # entry bar: only the stop can be judged safely after a touch fill
        while t < len(o):
            if (l[t] <= stop) if s > 0 else (h[t] >= stop):
                pnl += pos * (s * (stop - entry) / entry - 2 * fee); pos = 0; break
            if t > e and mode == 'half' and not half and ((h[t] >= x['tp1']) if s > 0 else (l[t] <= x['tp1'])):
                pnl += 0.5 * (s * (x['tp1'] - entry) / entry - 2 * fee); pos = 0.5; half = True
                stop = entry * (1 + s * 2 * fee); t += 1; continue
            tgt = x['tp1'] if mode == 'full' else x['tp2']
            if t > e and ((h[t] >= tgt) if s > 0 else (l[t] <= tgt)):
                pnl += pos * (s * (tgt - entry) / entry - 2 * fee); pos = 0; break
            t += 1
        if pos > 0: t = len(o) - 1; pnl += pos * (s * (c[t] - entry) / entry - 2 * fee)
        rows.append((df.time[e], pnl / unit, d)); busy = t
    return pd.DataFrame(rows, columns=['time', 'R', 'd'])


def summ(T):
    if len(T) < 20: return dict(n=len(T))
    days = max((T.time.iloc[-1] - T.time.iloc[0]).days, 1)
    g = 1 + RISK * T.R
    eq = np.cumprod(g.values); mon = g.groupby(T.time.dt.strftime('%Y-%m')).prod() - 1
    return dict(n=len(T), per_day=len(T) / days, win=(T.R > 0).mean(), avgR=T.R.mean(),
                R_month=T.R.sum() / (days / 30.4), mdd=(eq / np.maximum.accumulate(eq) - 1).min(),
                med_month=mon.median(), best_month=mon.max(), worst_month=mon.min(), med_stop=T.d.median())


if __name__ == '__main__':
    rows = []
    for ltf, htf in [('15min', '4h'), ('30min', '12h')]:
        df = bars(ltf)
        bias = htf_bias(htf, 3, df.time)
        for K in (3, 4, 6):
            for level in (0.5, 0.618, 0.705):
                S = setups(df, K, level)
                for use_bias, mode, fee in itertools.product((False, True), ('full', 'half'), (0.00035, 0.00025)):
                    T = run(df, S, bias if use_bias else None, mode, fee)
                    a = summ(T[T.time < SPLIT]); b = summ(T[T.time >= SPLIT])
                    rows.append(dict(ltf=ltf, htf=htf if use_bias else '-', K=K, level=level, mode=mode, fee=fee,
                                     **{f'{k}_in': v for k, v in a.items()}, **{f'{k}_out': v for k, v in b.items()}))
            print(ltf, K, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv('grid8_ict.csv', index=False)
    pd.set_option('display.width', 320)
    ok = R[(R.n_in >= 50) & (R.n_out >= 30)].copy(); ok['m'] = ok[['avgR_in', 'avgR_out']].min(axis=1)
    print('configs', len(R), 'positive both', (ok.m > 0).sum())
    cols = ['ltf', 'htf', 'K', 'level', 'mode', 'fee', 'n_in', 'win_in', 'avgR_in', 'n_out', 'per_day_out', 'win_out', 'avgR_out',
            'R_month_out', 'mdd_out', 'med_month_out', 'worst_month_out', 'med_stop_out']
    print(ok.sort_values('m', ascending=False)[cols].head(15).round(3).to_string(index=False))
