"""Trade the return to the volume-surge level.

SHORT setup (after a dump)          LONG setup (after a pump)
  quiet volume                        quiet volume
  -> first big-volume bar (surge)     -> first big-volume bar (surge)
  -> move >= K*ATR down               -> move >= K*ATR up
  -> bounce confirmed (1.5 ATR)       -> pullback confirmed (1.5 ATR)
  -> LIMIT SHORT at surge level       -> LIMIT LONG at surge level
     (wait up to WAIT bars)              (wait up to WAIT bars)
  stop   = level + STOP*ATR           stop   = level - STOP*ATR
  target = TGT: 'ext' (back to the low of the move) or an R multiple

No look-ahead: everything is decided on closed bars; the limit order only exists
after confirmation. Same-bar stop+target => counted as STOP. Entry bar that also
hits the stop => STOP.  Fees: entry maker 0.02%, target maker 0.02%,
stop taker 0.05% + 0.02% slippage.  Settings picked on <=2022 (or first 60% of
the data), judged once on the rest.
usage: python retr/surge_trade.py data/okx_btc-usdt-swap_5m.csv   [or data/btcusdt_1h.csv]"""
import numpy as np, pandas as pd, sys, itertools
src = sys.argv[1] if len(sys.argv) > 1 else 'data/btcusdt_1h.csv'
df = pd.read_csv(src, parse_dates=['time']).set_index('time')
h, l, c, v = (df[k].values for k in 'hlcv')
n = len(c); pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(100, min_periods=50).mean().values
vmed = pd.Series(v).rolling(720, min_periods=200).median().shift(1).values
MAKER, TAKER = 0.0002, 0.0007
CONF, MAXLEG = 1.5, 48

def events(K, M, QUIET):
    out, t, last = [], 800, 0
    while t < n - 300:
        ev = None
        for d in (-1, 1):
            win = c[t - 6:t]
            bi = t - 6 + (np.argmax(win) if d == -1 else np.argmin(win))
            if bi <= last or np.isnan(atr[bi]) or (c[t] - c[bi]) * d < K * atr[bi]: continue
            w0 = max(bi - 12, last + 1)
            q = v[w0 - 24:w0].mean()
            if QUIET and not q <= vmed[w0]: continue
            idx = [k for k in range(w0, t + 1) if v[k] >= M * q]
            if not idx: continue
            s = idx[0]; start = c[bi]
            ev = dict(d=d, t=t, s=s, level=c[s], start=start); break
        if not ev: t += 1; continue
        d, a = ev['d'], atr[ev['s']]
        ext, j = (l[t] if d == -1 else h[t]), t + 1
        while j < n and j - t <= MAXLEG:
            if d == -1:
                if l[j] < ext: ext = l[j]
                elif h[j] - ext >= CONF * a: break
            else:
                if h[j] > ext: ext = h[j]
                elif ext - l[j] >= CONF * a: break
            j += 1
        ev.update(ext=ext, conf=j, atr=a,
                  pos=abs(ev['level'] - ext) / max(abs(start - ext), 1e-9))
        out.append(ev); last = j; t = j + 1
    return out

def trade(ev, STOP, TGT, WAIT, MINPOS):
    d, L, a, ext, j = ev['d'], ev['level'], ev['atr'], ev['ext'], ev['conf']
    if ev['pos'] < MINPOS: return None
    side = d                                       # dump (-1) -> short, pump (+1) -> long
    # price already through the level at confirmation? then no clean limit fill
    if (side == -1 and c[j] >= L) or (side == 1 and c[j] <= L): return None
    stop = L - side * STOP * a
    tgt = ext if TGT == 'ext' else L + side * TGT * STOP * a
    risk = abs(stop - L) / L
    for k in range(j + 1, min(j + 1 + WAIT, n)):
        if (side == -1 and l[k] < ext) or (side == 1 and h[k] > ext): return None  # move resumed, cancel
        if (side == -1 and h[k] >= L) or (side == 1 and l[k] <= L):
            for m in range(k, min(k + 2000, n)):
                hit_s = (h[m] >= stop) if side == -1 else (l[m] <= stop)
                hit_t = (l[m] <= tgt) if side == -1 else (h[m] >= tgt)
                if m == k:  hit_t = False if not hit_s else hit_t   # entry bar: only stop counts
                if hit_s:
                    return dict(i=k, side=side, R=(-risk - MAKER - TAKER) / risk, risk=risk)
                if hit_t:
                    gain = abs(tgt - L) / L
                    return dict(i=k, side=side, R=(gain - 2 * MAKER) / risk, risk=risk)
            return None
    return None

split = df.index[int(n * 0.6)] if df.index[0].year >= 2019 else pd.Timestamp('2023-01-01', tz='UTC')
print(f'{src}: {n:,} bars {df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d}, train < {split:%Y-%m-%d}')
grid, cache = [], {}
for K, M, QUIET, STOP, TGT, WAIT, MINPOS in itertools.product(
        [3, 4, 6], [2, 3], [True, False], [0.5, 1.0, 2.0], ['ext', 1, 2], [12, 48], [0.0, 0.66]):
    key = (K, M, QUIET)
    if key not in cache: cache[key] = events(K, M, QUIET)
    T = [x for x in (trade(e, STOP, TGT, WAIT, MINPOS) for e in cache[key]) if x]
    if not T: continue
    X = pd.DataFrame(T); X['time'] = df.index[X.i]
    for side, nm in [(-1, 'SHORT'), (1, 'LONG')]:
        S = X[X.side == side]
        tr, te = S[S.time < split], S[S.time >= split]
        if len(tr) < 30: continue
        grid.append(dict(side=nm, K=K, M=M, quiet=QUIET, stop_atr=STOP, tgt=TGT, wait=WAIT, start_only=MINPOS > 0,
                         n_tr=len(tr), win_tr=(tr.R > 0).mean(), R_tr=tr.R.mean(),
                         n_te=len(te), win_te=(te.R > 0).mean() if len(te) else np.nan,
                         R_te=te.R.mean() if len(te) else np.nan))
G = pd.DataFrame(grid)
G.to_csv(src.replace('data/', 'retr/grid_').replace('.csv', '') + '.csv', index=False)
pd.set_option('display.width', 220)
for nm in ['SHORT', 'LONG']:
    S = G[G.side == nm].sort_values('R_tr', ascending=False)
    print(f'\n{nm}: {len(S)} settings tested. Best 8 chosen on TRAIN, then their TEST result:')
    print(S.head(8).round(3).to_string(index=False))
    print(f'  all settings: median train R {S.R_tr.median():+.3f}, median test R {S.R_te.median():+.3f}, '
          f'share of settings with test R > 0: {(S.R_te > 0).mean():.0%}')
