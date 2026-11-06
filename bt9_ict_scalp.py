"""ICT scalping: same rules as bt8_ict, on futures 1m data (last 3 years).
HTF -> LTF pairs: 15m -> 1m, 1h -> 5m. Stop = prior swing (영상 기준), sizing 10% margin.
Pick on 2023-11..2025-10, report 2025-11..2026-10 unseen."""
import itertools, numpy as np, pandas as pd
import bt8_ict as I
D = np.load('data/btc_fut_1m.npy')
F = pd.DataFrame(D[:, 1:], columns=list('ohlcv')); F['time'] = pd.to_datetime(D[:, 0], unit='ms', utc=True)
I.B5 = F.set_index('time'); I.SPLIT = pd.Timestamp('2025-11-01', tz='UTC')
rows = []
for ltf, htf in [('1min', '15min'), ('5min', '1h')]:
    df = I.bars(ltf); bias = I.htf_bias(htf, 3, df.time)
    for K in (3, 4, 6):
        for level in (0.5, 0.618, 0.705):
            S = I.setups(df, K, level)
            for use_bias, mode, fee in itertools.product((False, True), ('full', 'half'), (0.00035, 0.00025)):
                T = I.run(df, S, bias if use_bias else None, mode, fee)
                a = I.summ(T[T.time < I.SPLIT]); b = I.summ(T[T.time >= I.SPLIT])
                rows.append(dict(ltf=ltf, htf=htf if use_bias else '-', K=K, level=level, mode=mode, fee=fee,
                                 **{f'{k}_in': v for k, v in a.items()}, **{f'{k}_out': v for k, v in b.items()}))
        print(ltf, K, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv('grid9_ict_scalp.csv', index=False)
pd.set_option('display.width', 320)
ok = R[(R.n_in >= 50) & (R.n_out >= 30)].copy(); ok['m'] = ok[['avgR_in', 'avgR_out']].min(axis=1)
print('configs', len(R), 'positive both', (ok.m > 0).sum(), 'median out', ok.avgR_out.median().round(3))
cols = ['ltf', 'htf', 'K', 'level', 'mode', 'fee', 'n_in', 'win_in', 'avgR_in', 'n_out', 'per_day_out', 'win_out', 'avgR_out',
        'R_month_out', 'mdd_out', 'med_month_out', 'med_stop_out']
print(ok.sort_values('m', ascending=False)[cols].head(15).round(4).to_string(index=False))
