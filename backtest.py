"""Wave-3 setup backtest. Enter when wave 2 is confirmed finished.
Stop = start of wave 1 (count invalid there). Target = 1.618 x wave 1 from end of 2.
Fees 0.1%/side + 0.02% slippage. If stop and target hit in one bar: stop first."""
import sys, numpy as np, pandas as pd
from engine import atr, zigzag, internal_count, fib_score

df = pd.read_csv('data/btcusdt_1h.csv'); df['time'] = pd.to_datetime(df.time, format='ISO8601', utc=True)
A = atr(df)
FEE = 0.0012
K = float(sys.argv[1]) if len(sys.argv) > 1 else 8
KL = K / 2.5
TGT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.618
MINSCORE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
big = zigzag(df, K, A); low = zigzag(df, KL, A)
o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values

sig = []
for j in range(2, len(big)):
    p0, p1, p2 = big[j-2], big[j-1], big[j]
    up = p2[2] == -1                       # p0 low, p1 high, p2 low -> long
    s = 1 if up else -1
    w1 = s * (p1[1] - p0[1]); w2 = s * (p1[1] - p2[1])
    if w1 <= 0: continue
    r = w2 / w1
    if not (0.236 <= r < 1.0): continue    # rule: 2 never retraces all of 1
    known = p2[3]
    lowk = [p for p in low if p[3] <= known]           # causal
    if not internal_count(lowk, p0[0], p1[0], up, 5): continue   # 1 = five waves
    if not internal_count(lowk, p1[0], p2[0], up, 3): continue   # 2 = three waves
    sc = fib_score(r)
    if sc < MINSCORE: continue
    sig.append((known, up, p0[1], p2[1], w1, sc, r))

trades = []; busy = -1
for known, up, stop, base, w1, sc, r in sig:
    e = known + 1
    if e >= len(o) or e <= busy: continue
    s = 1 if up else -1
    entry = o[e]
    tgt = base + s * TGT * w1
    if s * (entry - stop) <= 0 or s * (tgt - entry) <= 0: continue
    res = None
    for t in range(e, len(o)):
        hit_s = l[t] <= stop if up else h[t] >= stop
        hit_t = h[t] >= tgt if up else l[t] <= tgt
        if hit_s: res = (t, stop); break
        if hit_t: res = (t, tgt); break
    if res is None: res = (len(o)-1, c[-1])
    t, x = res
    ret = s * (x - entry) / entry - FEE
    trades.append(dict(time=df.time[e], side='L' if up else 'S', entry=entry, exit=x,
                       ret=ret, bars=t-e, score=sc, r=r))
    busy = t

T = pd.DataFrame(trades)
if T.empty: print('no trades'); sys.exit()
eq = (1 + T.ret).cumprod()
dd = (eq / eq.cummax() - 1).min()
yrs = (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
print(f"K={K} tgt={TGT} min={MINSCORE} trades={len(T)} win={(T.ret>0).mean():.0%} "
      f"avg={T.ret.mean():+.2%} total={eq.iloc[-1]-1:+.0%} CAGR={eq.iloc[-1]**(1/yrs)-1:+.1%} "
      f"MDD={dd:.0%} PF={T.ret[T.ret>0].sum()/-T.ret[T.ret<0].sum():.2f}")
T['year'] = T.time.dt.year
print(T.groupby('year').ret.agg(['count', lambda x: (1+x).prod()-1]).round(3).to_string())
T.to_csv(f'trades_K{K}.csv', index=False)
