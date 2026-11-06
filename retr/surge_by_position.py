"""Split surge-level results by WHERE the surge bar sits inside the move.
pos = distance(surge close -> extreme) / distance(move start -> extreme)
pos ~1 : volume exploded at the very start (your yellow line case)
pos ~0 : volume exploded at the bottom/top (touching it again is trivial)"""
import numpy as np, pandas as pd, sys
df = pd.read_csv('data/btcusdt_1h.csv', parse_dates=['time']).set_index('time')
for tf in ['1h', '4h']:
    d_ = df if tf == '1h' else df.resample(tf).agg(dict(o='first', h='max', l='min', c='last', v='sum')).dropna()
    c = d_.c.values
    for M in [2, 3]:
        E = pd.read_csv(f'retr/surge_events_{tf}_M{M}.csv')
        start = np.array([c[max(s - 6, 0):t + 1].max() if d == -1 else c[max(s - 6, 0):t + 1].min()
                          for s, t, d in zip(E.s, E.t, E.d)])
        E['pos'] = (E['close'] - E.ext).abs() / (start - E.ext).abs()
        E['bin'] = pd.cut(E.pos, [-0.01, 0.33, 0.66, 1.01], labels=['bottom 1/3', 'middle 1/3', 'top 1/3 (start)'])
        for d, nm in [(-1, 'DUMP'), (1, 'PUMP')]:
            S = E[E.d == d]
            print(f'\n[{tf}, vol>={M}x] {nm}: where volume first exploded -> back to that close BEFORE new extreme (7d) | touch within 24h / 7d')
            for b, g in S.groupby('bin', observed=True):
                x = g.close_first_168.values.astype(float)
                rng = np.random.default_rng(0)
                bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
                print(f'   {b:<16} n={len(g):>3} | {x.mean():>4.0%} [{np.percentile(bs,2.5):.0%}-{np.percentile(bs,97.5):.0%}] | '
                      f'{g.close_touch_24.mean():>4.0%} / {g.close_touch_168.mean():>4.0%}')
