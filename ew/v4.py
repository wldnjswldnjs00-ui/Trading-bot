"""Elliott wave engine v4 — persistent multi-degree counting with every rule.

Each degree (1D, 4h, 1h, 15m, 5m by default) has main swings (K=3 ATR) and fine
swings (K=1 ATR). A main wave's internal structure is classified from the fine
swings inside it (impulse / diagonal / zigzag / flat / triangle / double / 3 /
complex), so "5-3-5" means real patterns one degree down, not just counts.

At every confirmed main swing the engine holds every count that still obeys the
rules — for every pattern and every position inside it — and scores it with the
guidelines. Counts persist: the same (start swing, pattern) is carried forward
and either advances one wave or dies with the rule that killed it.

Patterns and hard rules (Frost & Prechter):
  IMP  impulse 5 waves: 2 < 100% of 1; 3 never the shortest; 4 never enters 1's
       territory; 1, 3, 5 motive; 2, 4 corrective; 5 may truncate only if it is
       itself a 5 and 3 was strong.
  LD   leading diagonal (wave 1 / A): 5-3-5-3-5 or 3-3-3-3-3, 4 overlaps 1,
       wedge (contracting or expanding), 3 not the shortest.
  ED   ending diagonal (wave 5 / C): 3-3-3-3-3, overlap, wedge.
  ZZ   zigzag 5-3-5: B < 100% of A; C beyond A's end (truncation rare).
  FL   flat 3-3-5: B >= 90% of A; regular / expanded / running.
  TRI  triangle 3-3-3-3-3: contracting / barrier / expanding / running; only as
       wave 4, B, X or the last wave of a combination — never wave 2.
  DZ/TZ  double / triple zigzag: zigzags joined by X waves, each one deeper.
  DT/TT  double / triple three (combination): sideways, a triangle only last.
Guidelines scored: Fibonacci price, Fibonacci time, alternation (depth, form,
time), equality (5 ~ 1 when 3 extends), channelling (0-2 and 2-4 lines),
throw-over for ending diagonals, corrections ending in the prior 4th's area,
volume personality (3 loudest, 5 quieter), and agreement with the count one
degree up (top-down).
Every count states what the next wave must do (a level that proves it and a
level that kills it); the backtest grades that on price.
"""
import numpy as np, pandas as pd
from numba import njit
from ew.engine import load_1m, resample, _atr, _zigzag

TFS = ['1D', '4h', '1h', '15min', '5min']
K_MAIN, K_FINE = 3.0, 1.0
FIBP = np.array([0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0, 2.618])
MOTIVE = {'impulse', 'impulse_ext', 'diagonal'}
CORRECTIVE = {'zigzag', 'flat', 'triangle', 'double', 'three', 'complex'}


def fibscore(x, tol=0.08):
    return float(np.exp(-(np.min(np.abs(FIBP - x)) / tol) ** 2))


@njit(cache=True)
def first_touch(h, l, i0, hi, lo, maxbars):
    for i in range(i0, min(len(h), i0 + maxbars)):
        u = h[i] >= hi; d = l[i] <= lo
        if u and d: return 0, i - i0
        if u: return 1, i - i0
        if d: return -1, i - i0
    return 0, -1


class Deg:
    def __init__(self, tf, m1):
        b = resample(m1, tf); self.tf = tf
        self.h, self.l, self.c, self.o, self.v = b.h.values, b.l.values, b.c.values, b.o.values, b.v.values
        a = _atr(self.h, self.l, self.c, 14)
        self.ct = b.close_time.values.astype('datetime64[ns]').astype(np.int64)
        i, p, k, kn = _zigzag(self.h, self.l, a, K_MAIN); self.i, self.p, self.k, self.kn = i, p, k, self.ct[kn]
        fi, fp, fk, fkn = _zigzag(self.h, self.l, a, K_FINE); self.fi, self.fp, self.fk = fi, fp, fk
        self.cv = np.r_[0, np.cumsum(self.v)]
        self._cache = {}

    # ---- internal structure of a main wave from the fine swings inside it ----
    def structure(self, a, b):
        key = (a, b)
        if key in self._cache: return self._cache[key]
        i0, i1 = self.i[a], self.i[b]; s = 1 if self.p[b] > self.p[a] else -1
        lo = np.searchsorted(self.fi, i0, side='right'); hi = np.searchsorted(self.fi, i1, side='left')
        pts = [self.p[a]] + [self.fp[q] for q in range(lo, hi)] + [self.p[b]]
        # keep strict alternation
        clean = [pts[0]]
        for x in pts[1:]:
            if len(clean) >= 2 and (clean[-1] - clean[-2]) * (x - clean[-1]) > 0: clean[-1] = x
            else: clean.append(x)
        n = len(clean) - 1
        lab = 'mono'
        if n == 3:
            A, B, C = s * (clean[1] - clean[0]), s * (clean[1] - clean[2]), s * (clean[3] - clean[2])
            lab = 'zigzag' if A > 0 and B < A and s * (clean[3] - clean[1]) > 0 else 'flat' if B >= 0.9 * A else 'three'
        elif n == 5:
            w = [s * (clean[k + 1] - clean[k]) for k in range(5)]
            w1, w2, w3, w4, w5 = w[0], -w[1], w[2], -w[3], w[4]
            if min(w1, w2, w3, w4, w5) > 0 and w2 < w1 and not (w3 < w1 and w3 < w5):
                lab = 'impulse' if s * (clean[4] - clean[1]) > 0 else 'diagonal'
            elif all(abs(w[k + 1]) < abs(w[k]) for k in range(1, 4)):
                lab = 'triangle'
            else: lab = 'complex'
        elif n == 7: lab = 'double'
        elif n == 9:
            w = [s * (clean[k + 1] - clean[k]) for k in range(9)]
            lab = 'impulse_ext' if sum(x for x in w if x > 0) > 0 and s * (clean[-1] - clean[0]) > 0 else 'complex'
        elif n >= 11: lab = 'complex'
        self._cache[key] = lab
        return lab

    def vol(self, i0, i1):
        """average volume per bar between two bar indices"""
        return (self.cv[i1 + 1] - self.cv[i0]) / max(i1 - i0 + 1, 1)


def is_m(lab): return lab in MOTIVE or lab == 'mono'
def is_c(lab): return lab in CORRECTIVE or lab == 'mono'


# ---------------- pattern rule checks ----------------
# Each returns (alive, score, reason_if_dead, info). P = swing prices from the
# pattern start through the latest swing, s = direction of wave 1 / A / W,
# S = structure labels of the completed legs, D/idx for volume and time.
def chk_impulse(P, s, S, D, idx):
    k = len(P) - 1; L = [s * (P[q + 1] - P[q]) for q in range(k)]
    sc = []; info = {}
    if L[0] <= 0: return False, 0, 'wave 1 wrong way', info
    if not is_m(S[0]): return False, 0, 'wave 1 is not a 5', info
    if k >= 2:
        w2 = -L[1]
        if w2 <= 0 or w2 >= L[0]: return False, 0, 'wave 2 retraced all of wave 1', info
        if not is_c(S[1]): return False, 0, 'wave 2 is not corrective', info
        if S[1] == 'triangle': return False, 0, 'triangle in wave 2', info
        sc.append(fibscore(w2 / L[0]))
    if k >= 3:
        if L[2] <= 0 or s * (P[3] - P[1]) <= 0: return False, 0, 'wave 3 did not pass wave 1', info
        if not is_m(S[2]): return False, 0, 'wave 3 is not a 5', info
        sc.append(fibscore(L[2] / L[0])); sc.append(float(D.vol(idx[2], idx[3]) >= D.vol(idx[0], idx[1])))
    if k >= 4:
        w4 = -L[3]
        if w4 <= 0: return False, 0, 'wave 4 wrong way', info
        if s * (P[4] - P[1]) <= 0: return False, 0, 'wave 4 entered wave 1 territory', info
        if not is_c(S[3]): return False, 0, 'wave 4 is not corrective', info
        r2, r4 = -L[1] / L[0], w4 / L[2]
        sc.append(float((r2 > 0.5) != (r4 > 0.5)))                                    # alternation: depth
        sc.append(float((S[1] == 'zigzag') != (S[3] == 'zigzag')))                     # alternation: form
        t2, t4 = idx[2] - idx[1], idx[4] - idx[3]
        sc.append(float(t4 >= t2 or t2 >= 2 * t4))                                     # alternation: time
        # channel: line 0-2, parallel through 1 -> where 3 "should" end; reward 3 beyond it
        sc.append(fibscore(r4))
    if k >= 5:
        if L[4] <= 0: return False, 0, 'wave 5 wrong way', info
        if L[2] < L[0] and L[2] < L[4]: return False, 0, 'wave 3 is the shortest', info
        if not is_m(S[4]) and S[4] != 'diagonal': return False, 0, 'wave 5 is not a 5', info
        trunc = s * (P[5] - P[3]) < 0
        if trunc and not (S[4] in MOTIVE and L[2] >= 1.618 * L[0]): return False, 0, 'truncation without a strong 3', info
        ext = int(np.argmax([L[0], L[2], L[4]])) * 2 + 1; info['ext'] = ext; info['trunc'] = trunc
        if ext == 3: sc.append(float(0.8 <= L[4] / L[0] <= 1.25 or abs(L[4] / L[0] - 0.618) < 0.08))   # equality
        # 2-4 channel: parallel through 3 -> wave 5 target; score closeness
        t = idx; slope = (P[4] - P[2]) / max(t[4] - t[2], 1); tgt = P[3] + slope * (t[5] - t[3])
        sc.append(float(abs(P[5] - tgt) <= 0.15 * L[2]))
        sc.append(float(D.vol(idx[4], idx[5]) < D.vol(idx[2], idx[3])))                  # 5 quieter than 3
        tt = [idx[q + 1] - idx[q] for q in range(5)]
        sc.append(fibscore(tt[4] / max(tt[0], 1), 0.15))                                 # Fibonacci time
    return True, float(np.mean(sc)) if sc else 0.5, '', info


def chk_diagonal(P, s, S, D, idx, ending):
    k = len(P) - 1; L = [s * (P[q + 1] - P[q]) for q in range(k)]; sc = []
    if L[0] <= 0: return False, 0, 'wave 1 wrong way', {}
    need_m = not ending
    for q in range(k):
        if q % 2 == 0 and ending and not is_c(S[q]): return False, 0, 'ending diagonal wave is not a 3', {}
        if q % 2 == 0 and need_m and not (is_m(S[q]) or is_c(S[q])): return False, 0, 'bad leading diagonal wave', {}
        if q % 2 == 1 and not is_c(S[q]): return False, 0, 'diagonal 2/4 not corrective', {}
    if k >= 2 and -L[1] >= L[0]: return False, 0, 'wave 2 retraced all of wave 1', {}
    if k >= 3 and s * (P[3] - P[1]) <= 0: return False, 0, 'wave 3 did not pass wave 1', {}
    if k >= 4:
        if s * (P[4] - P[0]) <= 0: return False, 0, 'wave 4 beyond the start', {}
        contracting = L[2] < L[0] and -L[3] < -L[1]; expanding = L[2] > L[0] and -L[3] > -L[1]
        if not (contracting or expanding): return False, 0, 'not a wedge', {}
        sc.append(float(s * (P[4] - P[1]) <= 0))                                       # overlap is the norm
    if k >= 5:
        if L[2] < L[0] and L[2] < L[4]: return False, 0, 'wave 3 is the shortest', {}
        if (L[2] < L[0]) != (L[4] < L[2]): return False, 0, 'wedge shape broken by wave 5', {}
        slope = (P[3] - P[1]) / max(idx[3] - idx[1], 1); line = P[3] + slope * (idx[5] - idx[3])
        sc.append(float(s * (P[5] - line) > 0))                                        # throw-over
    if k >= 3: sc.append(fibscore(L[2] / L[0]))
    return True, float(np.mean(sc)) if sc else 0.5, '', {}


def chk_zigzag(P, s, S, D, idx):
    k = len(P) - 1; L = [s * (P[q + 1] - P[q]) for q in range(k)]; sc = []
    if L[0] <= 0: return False, 0, 'A wrong way', {}
    if not (is_m(S[0])): return False, 0, 'A is not a 5', {}
    if k >= 2:
        B = -L[1]
        if B <= 0 or B >= L[0]: return False, 0, 'B retraced all of A', {}
        if not is_c(S[1]): return False, 0, 'B is not corrective', {}
        sc.append(fibscore(B / L[0]))
    if k >= 3:
        if L[2] <= 0: return False, 0, 'C wrong way', {}
        if not (is_m(S[2]) or S[2] == 'diagonal'): return False, 0, 'C is not a 5', {}
        sc.append(float(s * (P[3] - P[1]) > 0)); sc.append(fibscore(L[2] / L[0]))
    return True, float(np.mean(sc)) if sc else 0.5, '', {}


def chk_flat(P, s, S, D, idx):
    k = len(P) - 1; L = [s * (P[q + 1] - P[q]) for q in range(k)]; sc = []; info = {}
    if L[0] <= 0: return False, 0, 'A wrong way', info
    if not is_c(S[0]): return False, 0, 'A is not a 3', info
    if k >= 2:
        B = -L[1]
        if B < 0.9 * L[0]: return False, 0, 'B under 90% of A', info
        if B > 1.618 * L[0]: return False, 0, 'B far beyond A start', info
        if not is_c(S[1]): return False, 0, 'B is not corrective', info
        info['kind'] = 'expanded' if B > 1.05 * L[0] else 'regular'
        sc.append(fibscore(B / L[0]))
    if k >= 3:
        if L[2] <= 0: return False, 0, 'C wrong way', info
        if not (is_m(S[2]) or S[2] == 'diagonal'): return False, 0, 'C is not a 5', info
        if info.get('kind') == 'expanded' and s * (P[3] - P[1]) < 0: info['kind'] = 'running'
        sc.append(fibscore(L[2] / L[0]))
    return True, float(np.mean(sc)) if sc else 0.5, '', info


def chk_triangle(P, s, S, D, idx, parent_pos):
    k = len(P) - 1; L = [abs(P[q + 1] - P[q]) for q in range(k)]; sc = []
    if parent_pos in ('IMP:2', 'LD:2'): return False, 0, 'triangle cannot be wave 2', {}
    if s * (P[1] - P[0]) <= 0: return False, 0, 'a wrong way', {}
    for q in range(k):
        if not is_c(S[q]): return False, 0, 'triangle leg is not a 3', {}
    if k >= 3:
        contracting = L[1] < L[0] and L[2] < L[1]; expanding = L[1] > L[0] and L[2] > L[1]; running = L[1] > L[0] and L[2] < L[1]
        if not (contracting or expanding or running): return False, 0, 'legs not triangular', {}
    if k >= 4 and not (L[3] < L[2] if L[2] < L[1] else L[3] > L[2] or L[1] > L[0]): return False, 0, 'd breaks the shape', {}
    if k >= 5:
        if not (L[4] < L[3] if L[3] < L[2] else L[4] > L[3]): return False, 0, 'e breaks the shape', {}
        sc.append(fibscore(L[4] / L[2])); sc.append(float(parent_pos.split(':')[-1] in ('4', 'B', 'X', 'Y', 'Z')))
    if k >= 3: sc.append(fibscore(L[2] / L[0]))
    return True, float(np.mean(sc)) if sc else 0.5, '', {}


def chk_double(P, s, S, D, idx, steep):
    """W-X-Y (k<=3) or W-X-Y-X-Z (k<=5). steep: double/triple zigzag; else combination."""
    k = len(P) - 1; L = [s * (P[q + 1] - P[q]) for q in range(k)]; sc = []
    if L[0] <= 0: return False, 0, 'W wrong way', {}
    for q in range(0, k, 2):
        if steep and S[q] not in ('zigzag', 'mono', 'three'): return False, 0, 'W/Y/Z is not a zigzag', {}
        if not steep and not is_c(S[q]): return False, 0, 'combination leg not corrective', {}
        if not steep and S[q] == 'triangle' and q != k - 1: return False, 0, 'triangle not last in combination', {}
    for q in range(1, k, 2):
        if not is_c(S[q]): return False, 0, 'X is not corrective', {}
        if -L[q] >= L[q - 1] * (1.0 if steep else 1.618): return False, 0, 'X too deep', {}
    for q in range(2, k, 2):
        if L[q] <= 0: return False, 0, 'Y/Z wrong way', {}
        if steep and s * (P[q + 1] - P[q - 1]) <= 0: return False, 0, 'Y/Z not deeper', {}
        sc.append(fibscore(L[q] / L[0]))
    return True, float(np.mean(sc)) if sc else 0.5, '', {}


PATTERNS = {'IMP': 5, 'LD': 5, 'ED': 5, 'ZZ': 3, 'FL': 3, 'TRI': 5, 'DZ': 3, 'TZ': 5, 'DT': 3, 'TT': 5}
NAMES = {'IMP': '12345', 'LD': '12345', 'ED': '12345', 'ZZ': 'ABC', 'FL': 'ABC', 'TRI': 'ABCDE',
         'DZ': 'WXY', 'TZ': 'WXYXZ', 'DT': 'WXY', 'TT': 'WXYXZ'}


def check(pat, P, s, S, D, idx, parent_pos):
    if pat == 'IMP': return chk_impulse(P, s, S, D, idx)
    if pat == 'LD': return chk_diagonal(P, s, S, D, idx, False)
    if pat == 'ED': return chk_diagonal(P, s, S, D, idx, True)
    if pat == 'ZZ': return chk_zigzag(P, s, S, D, idx)
    if pat == 'FL': return chk_flat(P, s, S, D, idx)
    if pat == 'TRI': return chk_triangle(P, s, S, D, idx, parent_pos)
    if pat in ('DZ', 'TZ'): return chk_double(P, s, S, D, idx, True)
    return chk_double(P, s, S, D, idx, False)


def expectation(pat, P, s):
    """What the next wave must do under this count: (direction, proof level, kill level, tag).
    proof = level the next move must reach, kill = level that ends the count first."""
    k = len(P) - 1; n = PATTERNS[pat]; last = P[-1]
    if k == n:                                                    # pattern complete -> the larger correction / resumption
        if pat in ('IMP', 'LD', 'ED'): return -s, P[4] if pat == 'IMP' else P[0], P[-1], 'complete:retrace'
        if pat == 'TRI': return -s, P[4], P[5], 'complete:thrust'
        return -s, P[0], P[-1], 'complete:resume'
    nxt_dir = s if k % 2 == 0 else -s                             # legs alternate
    if pat in ('IMP',) and k == 2: return s, P[1], P[0], 'wave3'
    if pat in ('IMP',) and k == 4: return s, P[3], P[1], 'wave5'
    if pat in ('LD',) and k == 2: return s, P[1], P[0], 'wave3'
    if pat in ('LD', 'ED') and k == 4: return s, P[3], P[0], 'wave5'
    if pat in ('ZZ', 'DZ', 'TZ') and k % 2 == 0: return s, P[k - 1], P[0], 'C/Y'
    if pat in ('FL', 'DT', 'TT') and k % 2 == 0: return s, P[k - 1] - s * 0.1 * abs(P[1] - P[0]), P[k] - s * 0.618 * abs(P[1] - P[0]), 'C/Y'
    if pat == 'TRI' and k < 5: return nxt_dir, last + nxt_dir * 0.5 * abs(P[k] - P[k - 1]), P[k - 1] if k >= 2 else P[0], 'tri-leg'
    # after a motive / A / W leg (or wave 1/3): a pullback of at least 38.2% before a new extreme
    seg = abs(P[k] - P[k - 1]); return nxt_dir, last + nxt_dir * 0.382 * seg, last, 'correction'
