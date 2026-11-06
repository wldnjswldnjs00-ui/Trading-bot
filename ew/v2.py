"""Elliott wave engine v2 — scenarios, consensus, volume profile, learned odds.

Per degree (1D, 4h, 1h, 15m, 5m, 1m) two zigzags run on the same bars:
  main (K=3 ATR)  -> the waves we count
  fine (K=1 ATR)  -> sub-waves inside each main wave (5 = motive, 3 = corrective,
                     7 = double zigzag / combination, 9 = extended impulse ...)
Every swing carries the moment it became known; nothing looks ahead.

At every newly confirmed main swing ("now") the engine lists every reading of
the last swings that the rules still allow, e.g. after low-high-higher-low:
  W12   waves 1-2 of an impulse      -> wave 3 next (up)
  ZAB   A-B of a zigzag (5-3)        -> wave C next (up)
  FAB   A-B of a flat (3-3, deep B)  -> wave C next (up, to about A's end)
  W34   waves 3-4 of an impulse      -> wave 5 next (up)
  TRI   inside a triangle            -> sideways (no direction)
  CNT   the up leg was a 3-wave correction in a bigger decline -> down next
Consensus = every surviving reading points the same way. The trade is the
owner's: enter now, stop just beyond the swing just made (prior low/high), half
off at the prior high/low, rest to 1.0 x the first leg (wave 3 / C ~ equality).

Features for the odds model: degree, which readings survive, consensus, bigger
degree direction, retrace zone (not exact ratios), volume node at the swing,
wick rejection, "pierced 0.618 but held on a volume node", sub-wave counts,
reward:risk. Odds are learned by logistic regression on past data only and
re-fit every year (walk-forward).
"""
import numpy as np, pandas as pd
from ew.engine import load_1m, resample, _atr, _zigzag

TFS = ['1D', '4h', '1h', '15min', '5min', '1min']
K_MAIN, K_FINE, BUF = 3.0, 1.0, 0.0005
FEE = 0.00035                                    # taker per side after rebate (OKX ~30%)
ZONES = [0, 0.382, 0.5, 0.618, 0.786, 1.0]


class Deg:
    def __init__(self, tf, m1):
        b = resample(m1, tf); self.tf = tf; self.b = b
        h, l, c = b.h.values, b.l.values, b.c.values; a = _atr(h, l, c, 14)
        self.ct = b.close_time.values.astype('datetime64[ns]').astype(np.int64)
        self.bt = b.time.values.astype('datetime64[ns]').astype(np.int64)
        i, p, k, kn = _zigzag(h, l, a, K_MAIN); self.i, self.p, self.k, self.kn = i, p, k, self.ct[kn]
        fi, fp, fk, fkn = _zigzag(h, l, a, K_FINE); self.fi, self.fkn = fi, self.ct[fkn]
        self.h, self.l, self.c, self.o, self.v = h, l, c, b.o.values, b.v.values

    def sub(self, i0, i1, now):
        """fine swings strictly inside bars (i0, i1) known by now, +1 = number of sub-waves."""
        a = np.searchsorted(self.fi, i0, side='right'); z = np.searchsorted(self.fi, i1, side='left')
        return int((self.fkn[a:z] <= now).sum()) + 1

    def leg_dir(self, now):
        j = np.searchsorted(self.kn, now, side='right') - 1
        return 0 if j < 0 else -int(self.k[j])


def vp_ratio(D, i_end, price, win=300, tol=0.003):
    a = max(0, i_end - win)
    if i_end - a < 30: return 1.0
    mid = (D.h[a:i_end] + D.l[a:i_end]) / 2; v = D.v[a:i_end]
    near = np.abs(mid / price - 1) <= tol
    span = (D.h[a:i_end].max() - D.l[a:i_end].min()) / price
    share_expected = min(1.0, 2 * tol / max(span, 1e-9))
    return float(v[near].sum() / max(v.sum() * share_expected, 1e-12))


def scenarios(D, j, now):
    """Readings of the swings ending at main swing j (the one just confirmed)."""
    p, i, s = D.p, D.i, 1 if D.k[j] == -1 else -1          # s = +1: last swing is a low -> next leg up?
    if j < 4: return None
    P = [p[j - 4], p[j - 3], p[j - 2], p[j - 1], p[j]]      # ... X0, X1, P0, P1, P2
    X0, X1, P0, P1, P2 = P
    L1 = s * (P1 - P0); L2 = s * (P1 - P2); r = L2 / L1 if L1 > 0 else 9
    sub1 = D.sub(i[j - 2], i[j - 1], now); sub2 = D.sub(i[j - 1], i[j], now)
    prev = s * (X1 - P0)                                    # leg before P0 (in -s direction if positive)
    sc = {}
    if L1 > 0 and 0 < r < 1:                                # higher low (s=+1) / lower high
        if sub1 >= 5 and sub1 % 2 == 1: sc['W12'] = s; sc['ZAB'] = s if r <= 0.886 else 0
        if sub1 == 3 and r >= 0.786: sc['FAB'] = s
        if sub1 == 3 and prev > 0 and prev > L1: sc['CNT'] = -s             # corrective up-leg inside a bigger decline
        # waves 3-4: X0 = 0, X1 = 1, P0 = 2, P1 = 3, P2 = 4
        w1 = s * (X1 - X0); w3 = L1; w2 = s * (X1 - P0)
        if w1 > 0 and w2 > 0 and w2 < w1 and s * (P2 - X1) > 0 and w3 > 0.618 * w1 and r < 0.618: sc['W34'] = s
        if L1 < abs(prev) and L2 < L1 and sub1 == 3 and sub2 == 3: sc['TRI'] = 0
    if not sc: return None
    dirs = {v for v in sc.values() if v != 0}
    zone = int(np.searchsorted(ZONES, min(r, 0.999), side='right') - 1)
    k_now = np.searchsorted(D.ct, now, side='left')
    if k_now + 1 >= len(D.c): return None
    px = D.c[k_now]; ib = i[j]
    rng = D.h[ib] - D.l[ib]
    wick = ((min(D.o[ib], D.c[ib]) - D.l[ib]) if s > 0 else (D.h[ib] - max(D.o[ib], D.c[ib]))) / rng if rng > 0 else 0
    vp = vp_ratio(D, i[j - 1], P2)
    return dict(s=s, P0=P0, P1=P1, P2=P2, L1=L1, r=r, zone=zone, sub1=sub1, sub2=sub2, sc=sc,
                consensus=int(len(dirs) == 1 and s in dirs), against=int(-s in dirs), px=px, k_now=k_now,
                wick=wick, vp=vp, pierced_618=int(r > 0.618), held_node=int(r > 0.618 and vp >= 1.5 and wick >= 0.4))


def outcome(D, x, maxbars=3000):
    """Enter at px now. Stop just beyond P2. TP1 = P1 (prior extreme). TP2 = P2 + 1.0 x L1.
    Returns (hit_tp1, hit_tp2 after tp1, R with half at TP1 + break-even, stop distance)."""
    s, px = x['s'], x['px']; stop = x['P2'] * (1 - s * BUF); tp1 = x['P1']; tp2 = x['P2'] + s * x['L1']
    d = s * (px - stop) / px
    if d <= 0 or s * (tp1 - px) <= 0: return None
    unit = d + 2 * FEE; h, l = D.h, D.l; k0 = x['k_now'] + 1; half = False; st = stop
    for t in range(k0, min(len(h), k0 + maxbars)):
        hs = (l[t] <= st) if s > 0 else (h[t] >= st)
        if hs:
            pnl = (0.5 * (s * (tp1 - px) / px - 2 * FEE) + 0.5 * (s * (st - px) / px - 2 * FEE)) if half else (s * (st - px) / px - 2 * FEE)
            return int(half), 0, pnl / unit, d, t - k0
        if not half and ((h[t] >= tp1) if s > 0 else (l[t] <= tp1)):
            half = True; st = px * (1 + s * 2 * FEE)
            continue
        if half and ((h[t] >= tp2) if s > 0 else (l[t] <= tp2)):
            pnl = 0.5 * (s * (tp1 - px) / px - 2 * FEE) + 0.5 * (s * (tp2 - px) / px - 2 * FEE)
            return 1, 1, pnl / unit, d, t - k0
    return None


def build(m1):
    rows = []
    degs = {tf: Deg(tf, m1) for tf in TFS}
    for g, tf in enumerate(TFS):
        D = degs[tf]; U = degs[TFS[g - 1]] if g > 0 else None
        for j in range(4, len(D.p)):
            now = D.kn[j]; x = scenarios(D, j, now)
            if x is None: continue
            o = outcome(D, x)
            if o is None: continue
            tp1, tp2, R, d, bars = o
            row = dict(tf=tf, time=pd.Timestamp(now, tz='UTC'), side=x['s'], tp1=tp1, tp2=tp2, R=R, stop_dist=d, bars=bars,
                       rr=abs(x['P1'] - x['px']) / abs(x['px'] - x['P2']), zone=x['zone'], sub1=x['sub1'], sub2=x['sub2'],
                       consensus=x['consensus'], against=x['against'], wick=x['wick'], vp=x['vp'], held_node=x['held_node'],
                       pierced_618=x['pierced_618'], parent=(U.leg_dir(now) * x['s']) if U else 0)
            for name in ('W12', 'ZAB', 'FAB', 'W34', 'CNT', 'TRI'): row[name] = int(name in x['sc'])
            rows.append(row)
        print(tf, 'signals', sum(1 for r in rows if r['tf'] == tf), flush=True)
    return pd.DataFrame(rows), degs


# ---------- odds model: L2 logistic regression, walk-forward ----------
def design(R):
    X = pd.DataFrame(index=R.index)
    for tf in TFS: X[f'tf_{tf}'] = (R.tf == tf).astype(float)
    for z in range(5): X[f'zone{z}'] = (R.zone == z).astype(float)
    for n in ('W12', 'ZAB', 'FAB', 'W34', 'CNT', 'TRI', 'consensus', 'against', 'held_node', 'pierced_618'): X[n] = R[n].astype(float)
    X['parent_with'] = (R.parent == 1).astype(float); X['parent_against'] = (R.parent == -1).astype(float)
    X['log_rr'] = np.log(R.rr.clip(0.05, 20)); X['wick'] = R.wick.clip(0, 1); X['log_vp'] = np.log(R.vp.clip(0.05, 20))
    X['sub1_5plus'] = (R.sub1 >= 5).astype(float); X['sub1_3'] = (R.sub1 == 3).astype(float); X['sub2_3'] = (R.sub2 == 3).astype(float)
    X['log_stop'] = np.log(R.stop_dist.clip(1e-4, 0.2))
    return X


def fit_logit(X, y, lam=1.0, iters=25):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w)); W = p * (1 - p)
        H = Z.T @ (Z * W[:, None]) + lam * np.eye(len(w)); g = Z.T @ (p - y) + lam * w
        w -= np.linalg.solve(H, g)
    return w, mu, sd


def predict_logit(m, X):
    w, mu, sd = m; Z = np.c_[np.ones(len(X)), (X - mu) / sd]; return 1 / (1 + np.exp(-Z @ w))


def walk_forward(R, years=range(2019, 2027)):
    R = R.copy(); R['year'] = R.time.dt.year
    X = design(R).values; out = []
    for Y in years:
        tr = (R.time < pd.Timestamp(f'{Y}-01-01', tz='UTC')).values; te = (R.year == Y).values
        if tr.sum() < 500 or te.sum() == 0: continue
        m1 = fit_logit(X[tr], R.tp1.values[tr]); m2 = fit_logit(X[tr & (R.tp1.values == 1)], R.tp2.values[tr & (R.tp1.values == 1)])
        mE = fit_ridge(X[tr], R.R.values[tr].clip(-1.5, 10))                      # expected R incl. fees
        P = R[te].copy(); P['p1'] = predict_logit(m1, X[te]); P['p2'] = predict_logit(m2, X[te]) * P.p1
        P['ev'] = predict_ridge(mE, X[te])
        out.append(P)
    return pd.concat(out)


def fit_ridge(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y); return w, mu, sd


def predict_ridge(m, X):
    w, mu, sd = m; return np.c_[np.ones(len(X)), (X - mu) / sd] @ w
