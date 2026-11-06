"""Owner idea: stop-loss also as a limit order right next to entry (near break-even).
Optimistic on purpose: the stop always fills at its price and pays maker.
Stop = entry -/+ eps; target = prior high (TP1) or the -0.62 extension."""
import itertools, numpy as np, pandas as pd
import bt8_ict as I, bt10_limit as L
M = L.MAKER
rows = []
for ltf, K, level in [('30min', 4, 0.705), ('30min', 6, 0.705), ('15min', 6, 0.705), ('30min', 3, 0.5), ('5min', 4, 0.705)]:
    df = I.bars(ltf); S, placed = L.orders(df, K, level)
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    for eps, tgt_key in itertools.product((0.001, 0.002, 0.003, 0.005), ('tp1', 'tp2')):
        out = []; busy = -1
        for x in S:
            e = x['e']; s = 1 if x['up'] else -1
            if e <= busy: continue
            entry = x['entry']; stop = entry * (1 - s * eps); tgt = x[tgt_key]
            unit = eps + 2 * M; t = e; px = None
            while t < len(o):
                if (l[t] <= stop) if s > 0 else (h[t] >= stop): px = stop; break
                if t > e and ((h[t] > tgt) if s > 0 else (l[t] < tgt)): px = tgt; break
                t += 1
            if px is None: t = len(o) - 1; px = c[t]
            out.append((df.time[e], (s * (px - entry) / entry - 2 * M) / unit)); busy = t
        T = pd.DataFrame(out, columns=['time', 'R']); T['d'] = eps
        for name, sub in (('in', T[T.time < I.SPLIT]), ('out', T[T.time >= I.SPLIT])):
            a = I.summ(sub)
            rows.append(dict(ltf=ltf, K=K, level=level, eps=eps, tgt=tgt_key, part=name, n=a.get('n'), win=a.get('win'), avgR=a.get('avgR')))
R = pd.DataFrame(rows).pivot_table(index=['ltf', 'K', 'level', 'eps', 'tgt'], columns='part', values=['n', 'win', 'avgR'])
pd.set_option('display.width', 250); print(R.round(3).to_string())
