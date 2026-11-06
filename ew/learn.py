"""Learn pattern probabilities from the past only, predict, grade, and explain misses.

Walk-forward by calendar year: to predict year Y, only patterns whose outcome was
settled before 1 Jan Y are used. Probability of "yes" for a pattern comes from
its bucket (degree, pattern, context, score, sub-waves known), shrunk toward the
same pattern pooled over all degrees (the fractal assumption) when the bucket
is thin.

Self-check:
  - Brier score vs the plain base rate (does knowing the pattern help at all?)
  - calibration: of the calls made at ~70%, did ~70% come true?
  - misses: which conditions are over-represented among confident misses
  - EV per call: p x reward:risk - (1 - p), in R, before fees
"""
import numpy as np, pandas as pd

TF_LEN = {'1W': '7D', '1D': '1D', '4h': '4h', '1h': '1h', '15min': '15min', '5min': '5min', '1min': '1min'}
M = 20                                                            # shrinkage strength toward the pooled pattern


def prep(R):
    R = R.copy(); R['time'] = pd.to_datetime(R.time, utc=True)
    R['settled'] = R.time + R.bars * R.tf.map(lambda x: pd.Timedelta(TF_LEN[x]))
    R['hi_score'] = R.score >= 0.5
    R['year'] = R.time.dt.year
    return R


def predict_year(R, Y, keys=('tf', 'pattern', 'ctx', 'hi_score', 'subs_known')):
    past = R[R.settled < pd.Timestamp(f'{Y}-01-01', tz='UTC')]; now = R[R.year == Y].copy()
    if len(past) < 200 or now.empty: return None
    base = past.ok.mean()
    pooled = past.groupby('pattern').ok.agg(['sum', 'size'])
    pooled_p = (pooled['sum'] + M * base) / (pooled['size'] + M)
    b = past.groupby(list(keys)).ok.agg(['sum', 'size']).reset_index()
    now = now.merge(b, on=list(keys), how='left').fillna({'sum': 0, 'size': 0})
    pp = now.pattern.map(pooled_p).fillna(base)
    now['p'] = (now['sum'] + M * pp) / (now['size'] + M)
    now['p_base'] = base
    return now


def report(R):
    R = prep(R); P = pd.concat([x for Y in range(2019, 2027) if (x := predict_year(R, Y)) is not None])
    out = []
    out.append(f'patterns {len(R):,}  graded predictions {len(P):,}  ({P.time.min():%Y-%m} .. {P.time.max():%Y-%m})')
    brier = ((P.p - P.ok) ** 2).mean(); brier0 = ((P.p_base - P.ok) ** 2).mean()
    out.append(f'Brier {brier:.4f} vs base rate {brier0:.4f}  -> skill {(1 - brier / brier0) * 100:+.1f}%')
    P['bin'] = (P.p * 10).clip(0, 9.99).astype(int) / 10
    cal = P.groupby('bin').agg(n=('ok', 'size'), said=('p', 'mean'), came_true=('ok', 'mean'))
    out.append('\nCALIBRATION (said vs came true)\n' + cal.round(3).to_string())
    by = P.groupby(['pattern', 'tf']).ok.mean().unstack('tf')[[t for t in TF_LEN if t in P.tf.unique()]]
    out.append('\nFRACTAL CHECK: success rate by pattern x degree\n' + by.round(2).to_string())
    ctx = P.groupby(['pattern', 'ctx']).ok.agg(['size', 'mean']).unstack('ctx')
    out.append('\nTOP-DOWN CONTEXT: success by context (with / against the bigger leg)\n' + ctx.round(3).to_string())
    conf = P[P.p >= 0.6]; miss = conf[~conf.ok]; hit = conf[conf.ok]
    if len(miss):
        rows = []
        for name, cond in [('against bigger leg', lambda d: d.ctx == 'against'), ('with bigger leg', lambda d: d.ctx == 'with'),
                           ('sub-waves unknown', lambda d: ~d.subs_known), ('low guideline score', lambda d: ~d.hi_score),
                           ('failed within 2 bars', lambda d: d.bars <= 2), ('tiny reward:risk < 0.5', lambda d: d.rr < 0.5)]:
            rows.append((name, cond(miss).mean(), cond(hit).mean()))
        L = pd.DataFrame(rows, columns=['condition', 'share_of_misses', 'share_of_hits']); L['lift'] = L.share_of_misses / L.share_of_hits
        out.append(f'\nMISSES among confident calls (p>=0.6): {len(miss):,} of {len(conf):,}\n' + L.round(3).to_string(index=False))
    P['ev'] = P.p * P.rr - (1 - P.p)
    P['ev_real'] = np.where(P.ok, P.rr, -1.0)
    sel = P[P.ev > 0.2]
    g = sel.groupby('tf').agg(n=('ev', 'size'), predicted_ev=('ev', 'mean'), realised_R=('ev_real', 'mean'), hit=('ok', 'mean'))
    out.append('\nTRADES the engine would take (predicted EV > +0.2R, before fees)\n' + g.round(3).to_string())
    return '\n'.join(out), P


if __name__ == '__main__':
    R = pd.read_csv('ew/patterns_btc.csv.gz')
    txt, P = report(R); print(txt)
    P.to_csv('ew/predictions_btc.csv.gz', index=False)
