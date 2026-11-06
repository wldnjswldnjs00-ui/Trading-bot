"""Run the v4 counter top-down over BTC and grade every count's own prediction.

Biggest degree first. Each degree's best count becomes the context for the next
degree down: if the bigger count is "in wave 3 up", a smaller count building an
up-impulse gets the agreement bonus; "in wave 2/4/B/X" favours a correction;
triangles are refused when the bigger count is in wave 2.

Logged:
  counts  — top-5 counts per swing with their prediction and how it turned out
  deaths  — every time a live count was killed, with the rule that killed it
Report: hit rate per pattern/position/degree, rank 1 vs the rest, effect of the
top-down agreement, calibration of the score, the rules that kill counts most,
and how often the best count changes.
"""
import sys, numpy as np, pandas as pd
from ew.v4 import Deg, TFS, PATTERNS, NAMES, check, expectation, first_touch
from ew.engine import load_1m

TOPN = 5


def parent_lookup(tl, now):
    if not tl: return 'none', 0
    j = np.searchsorted(tl[0], now, side='right') - 1
    return ('none', 0) if j < 0 else (tl[1][j], tl[2][j])


def run(tfs):
    m1 = load_1m(); counts, deaths = [], []; parent_tl = None
    for tf in tfs:
        D = Deg(tf, m1); alive_prev = {}; tl_t, tl_pos, tl_dir = [], [], []; prev_best = None; switches = 0
        for j in range(1, len(D.p)):
            now = D.kn[j]; ppos, pdir = parent_lookup(parent_tl, now)
            k_now = np.searchsorted(D.ct, now, side='left')
            if k_now + 1 >= len(D.c): break
            px = D.c[k_now]; alive = {}
            for pat, n in PATTERNS.items():
                for k in range(1, n + 1):
                    st = j - k
                    if st < 0: continue
                    P = list(D.p[st:j + 1]); s = 1 if P[1] > P[0] else -1
                    S = [D.structure(st + q, st + q + 1) for q in range(k)]
                    ok, sc, why, info = check(pat, P, s, S, D, D.i[st:j + 1], ppos)
                    key = (pat, st)
                    if not ok:
                        if key in alive_prev:
                            deaths.append((tf, pd.Timestamp(now, tz='UTC'), pat, NAMES[pat][k - 1], why))
                        continue
                    # top-down agreement: bigger wave in progress -> what this degree should be building
                    wave = ppos.split(':')[-1] if ppos != 'none' else ''
                    motive_parent = wave in ('1', '3', '5', 'A', 'C', 'W', 'Y', 'Z')
                    builds_motive = pat in ('IMP', 'LD', 'ED')
                    bonus = 0.0
                    if pdir != 0 and wave:
                        if motive_parent and builds_motive and s == pdir: bonus = 0.3
                        elif not motive_parent and not builds_motive and s == -pdir: bonus = 0.3
                        elif (builds_motive and s == -pdir) or (not builds_motive and s == pdir): bonus = -0.2
                    alive[key] = (pat, k, s, sc, bonus, sc + bonus + 0.04 * k, P, info)
            ranked = sorted(alive.values(), key=lambda x: -x[5])[:TOPN]
            if ranked:
                b = ranked[0]; n = PATTERNS[b[0]]
                in_prog = NAMES[b[0]][b[1]] if b[1] < n else 'post'
                tl_t.append(now); tl_pos.append(f'{b[0]}:{in_prog}'); tl_dir.append(b[2] if b[1] % 2 == 0 or b[1] == n else -b[2])
                ident = (b[0], j - b[1])
                if prev_best is not None and ident != prev_best: switches += 1
                prev_best = (b[0], j - b[1] + 1) if b[1] < n else None
                dirs = []
                for r, (pat, k, s, sc, bonus, tot, P, info) in enumerate(ranked):
                    d, proof, kill, tag = expectation(pat, P, s); dirs.append(d)
                    hi, lo = max(proof, kill), min(proof, kill)
                    if not (lo < px < hi): continue
                    res, bars = first_touch(D.h, D.l, k_now + 1, hi, lo, 3000)
                    if res == 0: continue
                    ok = (res == 1) == (proof > kill)
                    counts.append((tf, pd.Timestamp(now, tz='UTC'), r + 1, pat, NAMES[pat][k - 1], tag, d, sc, bonus, tot,
                                   ok, bars, abs(proof - px) / abs(kill - px), ppos, info.get('ext', 0), info.get('kind', '')))
                agree = len(set(dirs)) == 1
                if counts and counts[-1][1] == pd.Timestamp(now, tz='UTC'):
                    pass
            alive_prev = alive
        parent_tl = (np.array(tl_t), tl_pos, tl_dir)
        print(tf, 'swings', len(D.p), 'graded counts', sum(1 for c in counts if c[0] == tf), 'best-count switches', switches, flush=True)
    C = pd.DataFrame(counts, columns=['tf', 'time', 'rank', 'pattern', 'wave', 'tag', 'dir', 'score', 'bonus', 'total', 'ok', 'bars', 'rr',
                                      'parent', 'ext', 'kind'])
    X = pd.DataFrame(deaths, columns=['tf', 'time', 'pattern', 'last_wave', 'rule'])
    return C, X


def report(C, X):
    pd.set_option('display.width', 250); out = []
    out.append(f'graded predictions {len(C):,} | count deaths {len(X):,}')
    out.append('\nRANK 1 vs others (hit = next wave did what the count said)\n' +
               C.groupby(C['rank'].clip(upper=3)).ok.agg(['size', 'mean']).round(3).to_string())
    R1 = C[C['rank'] == 1]
    out.append('\nRANK-1 COUNTS: hit rate by pattern / wave just finished / degree\n' +
               R1.pivot_table(index=['pattern', 'wave', 'tag'], columns='tf', values='ok', aggfunc='mean').round(2).to_string())
    out.append('\nTOP-DOWN AGREEMENT (bonus +0.3 agrees, -0.2 disagrees with the bigger count)\n' +
               C.groupby('bonus').ok.agg(['size', 'mean']).round(3).to_string())
    C['sbin'] = (C.score * 5).clip(0, 4.99).astype(int) / 5
    out.append('\nGUIDELINE SCORE vs hit rate\n' + C.groupby('sbin').ok.agg(['size', 'mean']).round(3).to_string())
    out.append('\nRULES THAT KILL COUNTS (top 15)\n' + X.groupby(['pattern', 'rule']).size().sort_values(ascending=False).head(15).to_string())
    out.append('\nEXTENSIONS (completed impulses): which wave extended\n' + R1[R1.tag == 'complete:retrace'].groupby('ext').ok.agg(['size', 'mean']).round(3).to_string())
    out.append('\nFLAT KINDS\n' + C[C.pattern == 'FL'].groupby('kind').ok.agg(['size', 'mean']).round(3).to_string())
    return '\n'.join(out)


if __name__ == '__main__':
    tfs = sys.argv[1:] or TFS
    C, X = run(tfs); C.to_csv('ew/v4_counts_btc.csv.gz', index=False); X.to_csv('ew/v4_deaths_btc.csv.gz', index=False)
    txt = report(C, X); print(txt); open('logs/ew_v4_report.txt', 'w').write(txt)
