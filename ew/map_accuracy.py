"""How often does each "big wave" map point the right way, in real time?

Every map gives, at each moment, the direction of the wave it thinks is in
progress (+1 up / -1 down / 0 no call), using only swings already confirmed.
Truth = BTC's move over the next H days (beyond +-5%; smaller moves don't count).

Maps:
  D-best        daily best count (what full_map used)
  D-best-hold N daily best count, only after it has stayed the same count for N days
  D-vote X      share of all live daily counts pointing the same way >= X
  4h-best / 4h-vote X      the same on the 4h degree
  W+D           weekly best and daily best agree
"""
import numpy as np, pandas as pd
from ew.v4 import Deg, PATTERNS, check
from ew.v4_run import parent_lookup
from ew.engine import load_1m


def states(D, parent_tl=None):
    """Per confirmed swing: (time, best_dir, best_id, up_share, n_alive)."""
    out = []
    for j in range(1, len(D.p)):
        now = D.kn[j]; ppos, _ = parent_lookup(parent_tl, now) if parent_tl else ('none', 0)
        best = None; votes = []
        for pat, n in PATTERNS.items():
            for k in range(1, n + 1):
                st = j - k
                if st < 0: continue
                P = list(D.p[st:j + 1]); s = 1 if P[1] > P[0] else -1
                S = [D.structure(st + q, st + q + 1) for q in range(k)]
                ok, sc, _, _ = check(pat, P, s, S, D, D.i[st:j + 1], ppos)
                if not ok: continue
                # BIG trend this count implies: a motive pattern goes its own way, a correction goes against the trend
                trend = s if pat in ('IMP', 'LD', 'ED') else -s
                if k == n: trend = -trend if pat in ('IMP', 'LD', 'ED') else trend   # finished motive -> bigger correction next
                votes.append((trend, sc + 0.04 * k))
                if best is None or sc + 0.04 * k > best[0]: best = (sc + 0.04 * k, trend, (pat, st))
        if best is None: continue
        w = np.array([v[1] for v in votes]); d = np.array([v[0] for v in votes])
        out.append((now, best[1], best[2], float((w * (d > 0)).sum() / w.sum()), len(votes)))
    return pd.DataFrame(out, columns=['t', 'dir', 'id', 'up_share', 'n'])


def daily_series(S, days):
    """Value of each map on every day (state known before that day starts)."""
    S = S.copy(); S['t'] = pd.to_datetime(S.t, utc=True)
    idx = np.searchsorted(S.t.values, days.values, side='right') - 1
    ok = idx >= 0; out = pd.DataFrame(index=days)
    for c in ('dir', 'up_share'): out[c] = np.where(ok, S[c].values[np.clip(idx, 0, None)], np.nan)
    ids = S.id.values; out['id'] = [ids[i] if i >= 0 else None for i in idx]
    # how long the current best count has been unchanged (in days)
    since = []; last_id = None; start = None
    for d, i in zip(days, out.id):
        if i != last_id: last_id, start = i, d
        since.append((d - start).days if start is not None else 0)
    out['held'] = since
    return out


if __name__ == '__main__':
    m1 = load_1m()
    Dw, Dd, D4 = Deg('1W', m1), Deg('1D', m1), Deg('4h', m1)
    Sw = states(Dw); Sd = states(Dd); S4 = states(D4)
    close = m1.c.resample('1D').last().dropna()
    days = close.index[60:-40]
    mw, md, m4 = daily_series(Sw, days), daily_series(Sd, days), daily_series(S4, days)
    maps = {'D-best': md.dir}
    for N in (3, 7, 14): maps[f'D-best-hold{N}'] = md.dir.where(md.held >= N, 0)
    for X in (0.6, 0.7, 0.8): maps[f'D-vote{X}'] = np.where(md.up_share >= X, 1, np.where(md.up_share <= 1 - X, -1, 0))
    maps['4h-best'] = m4.dir
    for X in (0.6, 0.7, 0.8): maps[f'4h-vote{X}'] = np.where(m4.up_share >= X, 1, np.where(m4.up_share <= 1 - X, -1, 0))
    maps['W+D'] = np.where(mw.dir == md.dir, md.dir, 0)
    maps['W+D+4h'] = np.where((mw.dir == md.dir) & (md.dir == m4.dir), md.dir, 0)
    rows = []
    for H in (7, 14, 30):
        fwd = close.shift(-H) / close - 1; truth = np.sign(fwd.reindex(days)).where(fwd.reindex(days).abs() > 0.05, 0).values
        for name, m in maps.items():
            m = np.asarray(m, dtype=float); call = (m != 0) & ~np.isnan(m); scored = call & (truth != 0)
            rows.append(dict(horizon_days=H, map=name, coverage=call.mean(), accuracy=(m[scored] == truth[scored]).mean(),
                             n_days=int(scored.sum())))
    R = pd.DataFrame(rows); pd.set_option('display.width', 200)
    txt = R.pivot_table(index='map', columns='horizon_days', values=['accuracy', 'coverage']).round(3).to_string()
    txt = 'BIG-TREND ACCURACY OF BIG-WAVE MAPS (BTC 2017-2026, truth = next H days beyond +-5%)\n' + txt
    print(txt); open('logs/ew_map_accuracy_trend.txt', 'w').write(txt)
    pd.DataFrame(maps, index=days).to_csv('ew/maps_daily.csv')
