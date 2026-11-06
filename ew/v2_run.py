"""Run v2 on BTC: build signals, walk-forward odds, self-check report, current readings."""
import numpy as np, pandas as pd
from ew.v2 import build, walk_forward, design, fit_logit, predict_logit, fit_ridge, predict_ridge, TFS
from ew.engine import load_1m

if __name__ == '__main__':
    m1 = load_1m(); R, degs = build(m1); R.to_csv('ew/v2_signals_btc.csv.gz', index=False)
    P = walk_forward(R); P.to_csv('ew/v2_predictions_btc.csv.gz', index=False)
    pd.set_option('display.width', 250); out = []
    out.append(f'signals {len(R):,} | graded (2019+) {len(P):,}')
    b = P.tp1.mean(); brier = ((P.p1 - P.tp1) ** 2).mean(); b0 = ((b - P.tp1) ** 2).mean()
    out.append(f'P(reach prior high/low before stop): base {b:.3f} | Brier {brier:.4f} vs {b0:.4f} -> skill {(1 - brier / b0) * 100:+.1f}%')
    P['bin'] = (P.p1 * 10).clip(0, 9.99).astype(int) / 10
    out.append('\nCALIBRATION\n' + P.groupby('bin').agg(n=('tp1', 'size'), said=('p1', 'mean'), came_true=('tp1', 'mean')).round(3).to_string())
    for col, name in [('consensus', 'all readings agree (공통구간)'), ('against', 'a reading points the other way'),
                      ('held_node', 'pierced 0.618 but held on a volume node with a wick'), ('parent', 'bigger degree: +1 with / -1 against')]:
        out.append(f'\n{name}\n' + P.groupby(col).agg(n=('R', 'size'), tp1=('tp1', 'mean'), avgR=('R', 'mean')).round(3).to_string())
    out.append('\nRETRACE ZONE (0:<0.382 1:.382-.5 2:.5-.618 3:.618-.786 4:>.786)\n' +
               P.groupby('zone').agg(n=('R', 'size'), tp1=('tp1', 'mean'), avgR=('R', 'mean')).round(3).to_string())
    claims = {'W12': ('tp2', 'wave 3 runs >= wave 1'), 'ZAB': ('tp1', 'C goes past A end'), 'FAB': ('tp1', 'C reaches A end'),
              'W34': ('tp1', 'wave 5 goes past wave 3 end'), 'CNT': ('stop', 'decline resumes')}
    rows = []
    for k, (col, txt) in claims.items():
        S = P[P[k] == 1]
        if S.empty: continue
        hit = (S.tp1 == 0) if col == 'stop' else S[col] == 1
        rows.append((k, txt, len(S), hit.mean(), (S.tp1 == 0).mean()))
    out.append('\nSELF-CHECK: each reading\'s own claim\n' + pd.DataFrame(rows, columns=['reading', 'claim', 'n', 'came_true', 'stopped_first']).round(3).to_string(index=False))
    days = (P.time.max() - P.time.min()).days
    for th in (0.0, 0.1, 0.2, 0.3):
        S = P[P.ev > th]
        g = S.groupby('tf').agg(n=('R', 'size'), predicted=('ev', 'mean'), realised=('R', 'mean'), tp1=('tp1', 'mean'))
        g['per_day'] = g.n / days
        out.append(f'\nTRADES with predicted EV > {th:+.1f}R (fees included in realised R)\n' + g.round(3).to_string())
    by_year = P[P.ev > 0.1].groupby(P.time.dt.year).R.agg(['size', 'mean'])
    out.append('\nEV>0.1 by year\n' + by_year.round(3).to_string())
    # current readings with a model fitted on everything
    X = design(R).values; m1f = fit_logit(X, R.tp1.values); mEf = fit_ridge(X, R.R.values.clip(-1.5, 10))
    last = R.groupby('tf').tail(1).copy(); Xl = design(last).values
    last['p1'] = predict_logit(m1f, Xl); last['ev'] = predict_ridge(mEf, Xl)
    out.append('\nLATEST SIGNAL PER DEGREE (as of data end)\n' + last[['tf', 'time', 'side', 'W12', 'ZAB', 'FAB', 'W34', 'CNT', 'TRI', 'consensus',
                                                                    'zone', 'vp', 'p1', 'ev']].round(3).to_string(index=False))
    txt = '\n'.join(out); print(txt); open('logs/ew_v2_report.txt', 'w').write(txt)
