"""Which conditions make price come back to the volume box? (train vs test, tertiles from train)"""
import numpy as np, pandas as pd
E = pd.read_csv('retr/zone_events_5m.csv', parse_dates=['time'])
split = E.time.iloc[int(len(E) * 0.6)]
E['half'] = np.where(E.time < split, 'train', 'test')
NUM = {'imp_box': 'how far it ran (in box heights)', 'imp_atr': 'how far it ran (in ATR)',
       'vr': 'box volume vs before', 'box_atr': 'box height (ATR)', 'bounce': 'first bounce size',
       'dist_pct': 'box->extreme distance %'}
CAT = {'B': 'box length (bars)', 'trend': 'move WITH the 50h trend', 'sess': 'session', 'prelegs': 'had a first leg into the box'}
for k in NUM:
    q = E[E.half == 'train'][k].quantile([1 / 3, 2 / 3]).values
    E[k + '_b'] = np.select([E[k] <= q[0], E[k] <= q[1]], [f'low(<={q[0]:.2f})', f'mid(<={q[1]:.2f})'], f'high(>{q[1]:.2f})')
for d, nm in [(-1, 'DUMP -> back UP to box'), (1, 'PUMP -> back DOWN to box')]:
    S = E[E.d == d]
    print(f'\n======== {nm} (n={len(S)})   base: within 72h {S.back_72h.mean():.0%}, before new extreme {S.back_first.mean():.0%}')
    print(f"  {'condition':<45} {'within72h train/test':>22} {'before-new-extreme train/test':>30}   n train/test")
    for k, desc in list(NUM.items()) + list(CAT.items()):
        col = k + '_b' if k in NUM else k
        for val in sorted(S[col].unique(), key=str):
            X = S[S[col] == val]; a, b = X[X.half == 'train'], X[X.half == 'test']
            if len(a) < 25 or len(b) < 15: continue
            flag = ' <==' if (a.back_first.mean() - S.back_first.mean() > 0.08 and b.back_first.mean() - S.back_first.mean() > 0.08) else ''
            print(f"  {desc[:28] + ' = ' + str(val):<45} {a.back_72h.mean():>9.0%} / {b.back_72h.mean():<9.0%} {a.back_first.mean():>13.0%} / {b.back_first.mean():<12.0%} {len(a):>5}/{len(b):<4}{flag}")
