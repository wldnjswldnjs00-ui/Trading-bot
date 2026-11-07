"""Which conditions make 'retrace then break the extreme' a winning trade?
Single features and pairs, bins set on TRAIN (tertiles), judged on TEST.
A condition only counts if it is positive in train AND test AND both halves of
the test, with >= 60 test trades. Also a random-walk control for the break rate."""
import numpy as np, pandas as pd, itertools
E = pd.read_csv('retr/cont_events_5m.csv', parse_dates=['time'])
split = E.time.iloc[int(len(E) * 0.6)]
E['half'] = np.where(E.time < split, 'train', 'test')
mid = E[E.half == 'test'].time.median()
NUM = ['size', 'surge', 'quiet', 'dur', 'bvol', 'legpct', 'conf_retr']
CAT = ['trend', 'slope', 'sweep', 'sess']
for k in NUM:
    qs = E[E.half == 'train'][k].quantile([1 / 3, 2 / 3]).values
    E[k + '_b'] = np.select([E[k] <= qs[0], E[k] <= qs[1]], ['low', 'mid'], 'high')
FEATS = [k + '_b' for k in NUM] + CAT
TR = [c for c in E.columns if c.startswith('R_')]

# random-walk control: same distance from extreme (in ATR) at random times
df = pd.read_csv('data/okx_btc_5m_trim.csv')
l, h, c = df.l.values, df.h.values, df.c.values
rng = np.random.default_rng(0)
pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100).mean().values
dist = (E.conf_retr * E['size']).values          # distance to extreme in ATR
ii = rng.integers(500, len(c) - 300, 20000); dd = rng.choice(dist, 20000)
lvl = c[ii] - dd * atr[ii]
hit = np.array([(l[i + 1:i + 289] < lv).any() for i, lv in zip(ii, lvl)])
print(f'CONTROL: random moments, a level the same ATR distance below price -> broken within 24h: {hit.mean():.0%}')
print(f'ACTUAL : dumps {E[E.d == -1].brk.mean():.0%}, pumps {E[E.d == 1].brk.mean():.0%}\n')

out = []
for d in (-1, 1):
    S = E[E.d == d]
    conds = [((f,), (v,)) for f in FEATS for v in S[f].unique()]
    conds += [((f, g), (a, b)) for f, g in itertools.combinations(FEATS, 2)
              for a in S[f].unique() for b in S[g].unique()]
    for fs, vs in conds:
        m = np.ones(len(S), bool)
        for f_, v_ in zip(fs, vs): m &= (S[f_] == v_).values
        X = S[m]
        for col in TR:
            tr = X[X.half == 'train'][col].dropna(); te = X[X.half == 'test'][col].dropna()
            if len(tr) < 100 or len(te) < 60: continue
            te_t = X[X.half == 'test']
            t1 = te_t[te_t.time < mid][col].dropna(); t2 = te_t[te_t.time >= mid][col].dropna()
            out.append(dict(side='SHORT' if d == -1 else 'LONG', cond=' & '.join(f'{a}={b}' for a, b in zip(fs, vs)),
                            trade=col[2:], n_tr=len(tr), R_tr=tr.mean(), n_te=len(te), R_te=te.mean(),
                            R_te1=t1.mean(), R_te2=t2.mean(), win_te=(te > 0).mean(),
                            brk_te=X[X.half == 'test'].brk.mean()))
O = pd.DataFrame(out)
O.to_csv('retr/conditions_all.csv', index=False)
print(f'tested {len(O):,} (condition x trade) combos')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
for sd in ['SHORT', 'LONG']:
    X = O[O.side == sd]
    print(f'\n{sd}: top 10 by TRAIN R, and what they did on TEST')
    print(X.sort_values('R_tr', ascending=False).head(10).round(3).to_string(index=False))
    ok = X[(X.R_tr > 0) & (X.R_te > 0) & (X.R_te1 > 0) & (X.R_te2 > 0)]
    print(f'  survivors (train>0, test>0, both test halves>0): {len(ok)} of {len(X)}  '
          f'(by luck alone expect ~{len(X) * (X.R_tr > 0).mean() * 0.125:.0f} if there is no edge)')
    if len(ok): print(ok.sort_values('R_te', ascending=False).head(10).round(3).to_string(index=False))
