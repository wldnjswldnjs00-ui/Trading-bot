"""Full pattern catalogue with fine sub-wave counts, graded by each pattern's own
textbook aftermath, and compared with the same windows left unlabelled.

Sub-wave counts come from the fine zigzag (K=1) inside each main (K=3) wave.
Labels (motive = odd count >= 5, corrective = 3; 7 = double / combination):
  impulse (+ which wave extended), truncated 5th,
  leading diagonal (5-3-5-3-5, overlap, wedge), ending diagonal (3-3-3-3-3, wedge),
  zigzag (5-3-5), flat regular / expanded / running (3-3-5),
  double zigzag (3-3-3 steep), double three / combination (3-3-3 sideways),
  triangle contracting / barrier / expanding / running (3-3-3-3-3).
Aftermath question per label (first level touched after the pattern is known):
  impulse, truncated   -> back to the wave-4 end before a new extreme
  ending diagonal      -> back to the diagonal start before a new extreme
  leading diagonal     -> beyond the diagonal end before below its start
  zigzag/flat/double/combination -> beyond the correction start before beyond its end
  triangle             -> beyond the d end before beyond the e end
LIFT = hit rate of the label / hit rate of every unlabelled window asking the
same question on the same degree.
"""
import sys, numpy as np, pandas as pd
from ew.v2 import Deg, TFS
from ew.engine import load_1m
from ew.scan import first_touch

TOL = 0.06


def motive(n): return n is not None and n >= 5 and n % 2 == 1
def corr3(n): return n == 3


def label6(P, s, sub):
    L = [s * (P[i + 1] - P[i]) for i in range(5)]
    w1, w2, w3, w4, w5 = L[0], -L[1], L[2], -L[3], L[4]
    out = []
    if min(w1, w2, w3, w4, w5) > 0 and w2 < w1:
        overlap = s * (P[4] - P[1]) <= 0
        if not overlap and not (w3 < w1 and w3 < w5) and motive(sub[0]) and motive(sub[2]) and motive(sub[4]):
            ws = [w1, w3, w5]; e = int(np.argmax(ws)); others = [x for k, x in enumerate(ws) if k != e]
            ext = f'_ext{2 * e + 1}' if ws[e] >= 1.618 * max(others) else ''
            out.append('impulse' + ext)
        if overlap and s * (P[4] - P[0]) > 0 and not (w3 < w1 and w3 < w5):
            contracting = w3 < w1 and w5 < w3 and w4 < w2; expanding = w3 > w1 and w5 > w3 and w4 > w2
            if contracting or expanding:
                shape = 'contracting' if contracting else 'expanding'
                if all(motive(sub[k]) for k in (0, 2, 4)) and corr3(sub[1]) and corr3(sub[3]): out.append(f'diagonal_leading_{shape}')
                elif all(corr3(x) for x in sub): out.append(f'diagonal_ending_{shape}')
    # truncated fifth: 5 legs, wave 5 a motive 5 that fails to pass wave 3's end
    if min(w1, w2, w3, w4) > 0 and w5 > 0 and s * (P[5] - P[3]) < 0 and s * (P[4] - P[1]) > 0 and motive(sub[4]) and motive(sub[2]):
        out.append('truncated_5th')
    # triangles: alternating legs, all 3-wave
    a, b, c, d, e = [abs(x) for x in L]
    if all(corr3(x) for x in sub) and L[0] * L[1] < 0:
        if b < a and c < b * 1.0 + a and c < a and d < b and e < c:
            barrier = abs(P[2] - P[4]) / max(abs(P[1] - P[2]), 1e-12) < TOL or abs(P[1] - P[3]) / max(abs(P[1] - P[2]), 1e-12) < TOL
            running = b > a
            out.append('triangle_barrier' if barrier else 'triangle_contracting')
        elif b > a and c < b and d < c and e < d:
            out.append('triangle_running')
        elif c > a and d > b and e > c:
            out.append('triangle_expanding')
    return out


def label4(P, s, sub):
    A, B, C = s * (P[1] - P[0]), s * (P[1] - P[2]), s * (P[3] - P[2])
    if min(A, B, C) <= 0: return []
    r, q = B / A, C / A; beyondA = s * (P[3] - P[1]) > 0; out = []
    if motive(sub[0]) and corr3(sub[1]) and motive(sub[2]) and r <= 0.886 and beyondA: out.append('zigzag')
    if corr3(sub[0]) and corr3(sub[1]) and motive(sub[2]) and r >= 0.786:
        if r > 1.0 and beyondA: out.append('flat_expanded')
        elif r > 1.0 and not beyondA: out.append('flat_running')
        else: out.append('flat_regular')
    if corr3(sub[0]) and corr3(sub[2]) and sub[1] is not None and sub[1] >= 3 and r < 1.0:
        out.append('double_zigzag' if beyondA and q >= 0.618 else 'double_three')
    return out


def question(lbl):
    if lbl.startswith(('impulse', 'truncated')): return 'back_to_w4'
    if lbl.startswith('diagonal_ending'): return 'back_to_start'
    if lbl.startswith('diagonal_leading'): return 'beyond_end'
    if lbl.startswith('triangle'): return 'thrust'
    return 'resume'


def levels(q, P, s):
    """(yes level, no level); for every question the two levels bracket price at decision time."""
    if q == 'back_to_w4': return P[4], P[5]
    if q == 'back_to_start': return P[0], P[5]
    if q == 'beyond_end': return P[5] + s * 1e-9, P[0]          # long side for s=+1: yes above, no below
    if q == 'thrust': return P[4], P[5]
    return P[0], P[3]


def run(tfs):
    m1 = load_1m(); rows = []
    for tf in tfs:
        D = Deg(tf, m1); h, l, c = D.h, D.l, D.c
        for j in range(5, len(D.p)):
            now = D.kn[j]; k_now = np.searchsorted(D.ct, now, side='left')
            if k_now + 1 >= len(c): continue
            px = c[k_now]
            for n in (4, 6):
                sl = slice(j - n + 1, j + 1); P = list(D.p[sl]); I = D.i[sl]
                s = 1 if D.k[j - n + 1] == -1 else -1
                sub = [D.sub(I[k], I[k + 1], now) for k in range(n - 1)]
                labels = label6(P, s, sub) if n == 6 else label4(P, s, sub)
                qs = ['back_to_w4', 'back_to_start', 'beyond_end', 'thrust'] if n == 6 else ['resume']
                for q in qs:
                    yes, no = levels(q, P, s)
                    hi, lo = max(yes, no), min(yes, no)
                    if not (lo < px < hi): continue
                    res, bars = first_touch(h, l, k_now + 1, hi, lo, 3000)
                    if res == 0: continue
                    ok = (res == 1) == (yes > no)
                    mine = [x for x in labels if question(x) == q]
                    for lbl in (mine or ['(unlabelled)']):
                        rows.append((tf, lbl, q, pd.Timestamp(now, tz='UTC'), s, ok, bars, '-'.join(str(x) for x in sub)))
        print(tf, len(rows), flush=True)
    return pd.DataFrame(rows, columns=['tf', 'label', 'question', 'time', 'dir', 'ok', 'bars', 'subs'])


if __name__ == '__main__':
    tfs = sys.argv[1:] or TFS
    R = run(tfs); R.to_csv('ew/v3_patterns_btc.csv.gz', index=False)
    base = R[R.label == '(unlabelled)'].groupby(['tf', 'question']).ok.mean().rename('base')
    L = R[R.label != '(unlabelled)'].groupby(['label', 'question', 'tf']).ok.agg(n='size', hit='mean').reset_index()
    L = L.merge(base.reset_index(), on=['tf', 'question'], how='left'); L['lift'] = L.hit / L.base
    pooled = R[R.label != '(unlabelled)'].groupby(['label', 'question']).ok.agg(n='size', hit='mean').reset_index()
    pb = R[R.label == '(unlabelled)'].groupby('question').ok.mean()
    pooled['base'] = pooled.question.map(pb); pooled['lift'] = pooled.hit / pooled.base
    pd.set_option('display.width', 220)
    txt = ('ALL DEGREES POOLED\n' + pooled.sort_values('lift', ascending=False).round(3).to_string(index=False) +
           '\n\nBY DEGREE (n>=30)\n' + L[L.n >= 30].pivot_table(index='label', columns='tf', values='lift').round(2).to_string() +
           '\n\nCOUNTS BY DEGREE\n' + L.pivot_table(index='label', columns='tf', values='n', aggfunc='sum').fillna(0).astype(int).to_string())
    print(txt); open('logs/ew_v3_patterns_report.txt', 'w').write(txt)
