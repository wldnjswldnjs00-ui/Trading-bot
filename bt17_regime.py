"""Big-degree BTC regime (known at the time) as a direction filter.

Regime from BTC daily bars (2017-2026, from the 1h archive): zigzag on daily
bars, last four confirmed swings -> HH+HL = bull, LH+LL = bear, else range.
A swing is only known once price has turned by K x ATR(50d), so the regime
switches late on purpose — exactly what a live bot would see.
Then the ICT 30m strategy on 150 coins is re-run recording long/short, and
every trade is tagged with the BTC regime at entry.
"""
import glob, os, sys, numpy as np, pandas as pd
from engine import atr, zigzag
import bt16_four as F
from bt13_ict_cvd import load30, SPLIT


def btc_daily():
    d = pd.read_csv('data/btcusdt_1h.csv'); d['time'] = pd.to_datetime(d.time, format='ISO8601', utc=True)
    return d.set_index('time').resample('1D').agg(dict(o='first', h='max', l='min', c='last', v='sum')).dropna().reset_index()


def regime_series(D, K):
    piv = zigzag(D, K, atr(D, 50)); b = F.struct_bias(piv, len(D))
    known = (D.time + pd.Timedelta('1D'))                        # daily bar usable after it closes
    return pd.Series(b, index=known.values), piv


if __name__ == '__main__':
    D = btc_daily()
    regs = {K: regime_series(D, K) for K in (2.0, 3.0, 4.0)}
    for K, (r, piv) in regs.items():
        ch = r[r.diff() != 0]
        print(f'K={K}: switches {len(ch)}  bull days {(r == 1).mean():.0%} bear {(r == -1).mean():.0%} range {(r == 0).mean():.0%}')
        print('   ', ' '.join(f"{t:%Y-%m}:{'B' if v == 1 else 'S' if v == -1 else '-'}" for t, v in ch.items()))
    syms = [s for s in sorted({os.path.basename(f).split('USDT')[0] for f in glob.glob('data/alt5m/*.zip')})
            if s in set(open(sys.argv[1]).read().split())] if len(sys.argv) > 1 else ['BTC']
    rows = []
    for sym in syms:
        try: df = F.load30(sym) if hasattr(F, 'load30') else load30(sym)
        except Exception: continue
        if len(df) < 2000: continue
        P = F.prep(df); setups = {'ict': F.ict_setups(P), 'sweep': F.sweep_setups(P)}
        for e, py in (('ict', False), ('ict', True), ('sweep', False), ('sweep', True)):
            busy = -1
            for x in setups[e]:
                if x['start'] <= busy: continue
                r = F.simulate(P, x, py, False, False)
                if r is None: continue
                ft, tend, R, adds = r; busy = tend
                rows.append((f'{e}|{"pyr" if py else "half_trail"}', sym, df.time[ft], 1 if x['up'] else -1, R))
        print(sym, flush=True)
    T = pd.DataFrame(rows, columns=['cfg', 'sym', 'time', 'side', 'R'])
    for K, (r, _) in regs.items():
        idx = np.searchsorted(r.index.values, T.time.values.astype('datetime64[ns]'), side='right') - 1
        T[f'reg{K}'] = np.where(idx >= 0, r.values[np.clip(idx, 0, None)], 0)
    T.to_csv('regime_trades.csv.gz', index=False)
    days = (T.time.max() - T.time.min()).days
    pd.set_option('display.width', 250)
    for K in regs:
        T['al'] = np.where(T[f'reg{K}'] == 0, 'range', np.where(T[f'reg{K}'] == T.side, 'with', 'against'))
        G = T.groupby(['cfg', 'al', 'side']).R.agg(n='size', avgR='mean', win=lambda x: (x > 0).mean())
        G['per_day'] = G.n / days
        print(f'\n=== BTC daily regime K={K} ===\n', G.round(3).to_string())
        W = T[T['al'] == 'with']
        S = W.groupby('cfg').agg(n=('R', 'size'), avgR=('R', 'mean'),
                                 first=('R', lambda x: x[W.loc[x.index, 'time'] < SPLIT].mean()),
                                 last=('R', lambda x: x[W.loc[x.index, 'time'] >= SPLIT].mean()))
        S['per_day'] = S.n / days; S['R_month'] = S.avgR * S.per_day * 30.4
        print('trend-aligned only:\n', S.round(3).to_string())
