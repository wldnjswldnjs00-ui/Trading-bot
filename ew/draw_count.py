"""Draw the Elliott count from the 2025-10 ATH to now: big / medium / small degree + scenarios."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
SURF, INK, INK2, GRID = '#1a1a19', '#f0efec', '#c3c2b7', '#33332f'
BIG, MED, SMALL, BEAR, BULL = '#3987e5', '#d95926', '#199e70', '#e66767', '#9085e9'
UP, DN = '#8a8984', '#4d4c48'
H = pd.read_csv('/tmp/claude-0/-home-user-Trading-bot/08951856-c9aa-531b-a769-1eb25b35b6ce/scratchpad/btc_1h_full.csv', parse_dates=['time']).set_index('time')
M5 = pd.read_csv('data/okx_btc-usdt-swap_5m.csv', parse_dates=['time']).set_index('time')
agg = dict(o='first', h='max', l='min', c='last')
T = lambda s: pd.Timestamp(s, tz='UTC')

def candles(ax, d, w):
    up = d.c >= d.o
    for m, col in [(up, UP), (~up, DN)]:
        x = d[m]
        ax.vlines(x.index, x.l, x.h, color=col, lw=0.6, zorder=1)
        ax.bar(x.index, (x.c - x.o).abs().clip(lower=x.c * 1e-4), bottom=np.minimum(x.o, x.c), width=w, color=col, zorder=1, lw=0)

def waves(ax, pts, col, lw, fs, above):
    xs, ys = [T(p[0]) for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=col, lw=lw, zorder=3, solid_joinstyle='round')
    ax.scatter(xs, ys, s=18, color=col, zorder=4, edgecolors=SURF, linewidths=1.5)
    for (t, p, lab), up in zip(pts, above):
        if not lab: continue
        ax.annotate(lab, (T(t), p), xytext=(0, 9 if up else -9), textcoords='offset points', ha='center',
                    va='bottom' if up else 'top', color=INK, fontsize=fs, fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc=SURF, ec=col, lw=1))

def style(ax, title):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8); ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k'))
    ax.set_title(title, color=INK, fontsize=11, loc='left', fontweight='bold')

def level(ax, y, txt, col, x0=None):
    ax.axhline(y, color=col, lw=0.8, ls=':', zorder=2)
    ax.text(ax.get_xlim()[0] if x0 is None else x0, y, f' {txt}', color=INK2, fontsize=7.5, va='bottom', zorder=5)

fig = plt.figure(figsize=(15, 17), facecolor=SURF)
gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1, 1], hspace=0.22, top=0.88)

# ---------- 1. BIG degree (daily) ----------
ax = fig.add_subplot(gs[0]); d = H[H.index >= '2025-07-01'].resample('1D').agg(agg).dropna()
candles(ax, d, 0.7)
big = [('2025-10-06 18:00', 126200, 'ATH'), ('2025-10-10 21:00', 102000, '1'), ('2025-10-27 07:00', 116400, '2'),
       ('2025-11-21 12:00', 80600, '3'), ('2026-01-14 20:00', 97924, '4'), ('2026-02-06 00:00', 60000, '5 = (A)'),
       ('2026-05-06 11:00', 82850, 'a'), ('2026-07-01 01:00', 57800, 'b'), ('2026-09-21 20:00', 87396, 'c = (B)'),
       ('2026-11-06 15:00', 74800, '(C)?')]
waves(ax, big, BIG, 2, 9, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
fut = pd.date_range('2026-11-07', '2027-04-30', freq='D', tz='UTC')
ax.plot([T('2026-11-06 15:00'), T('2026-11-25'), T('2027-04-15')], [74800, 79000, 54300], color=BEAR, lw=1.5, ls='--', zorder=3)
ax.plot([T('2026-11-06 15:00'), T('2026-11-20'), T('2027-03-15')], [74800, 72600, 98000], color=BULL, lw=1.5, ls='--', zorder=3)
ax.text(T('2027-04-15'), 54300, ' main: (C) → 54k–46.5k', color=INK, fontsize=8.5, va='center')
ax.text(T('2027-03-15'), 98000, ' alt: wave 3 up', color=INK, fontsize=8.5, va='center')
ax.set_xlim(T('2025-07-01'), T('2027-06-15')); ax.set_ylim(44000, 130000)
for y, t, c in [(57800, '57.8k  bull count dies below', BEAR), (46500, '46.5k  (C)=0.618×(A)', BEAR),
                (87396, '87.4k  break = bull count', BULL)]:
    level(ax, y, t, c, T('2026-12-20'))
style(ax, 'BIG degree (daily)  ·  main: 126.2k ATH → (A) 5-wave down → (B) expanded flat → (C) in progress')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

# ---------- 2. MEDIUM degree (4h) ----------
ax = fig.add_subplot(gs[1]); d = H[H.index >= '2026-09-12'].resample('4h').agg(agg).dropna()
candles(ax, d, 0.12)
med = [('2026-09-21 20:00', 87396, '(B) top'), ('2026-10-15 13:00', 79800, '1'), ('2026-10-21 16:00', 84898, '2'),
       ('2026-11-04 03:00', 75690, '3'), ('2026-11-04 15:15', 81200, '4'), ('2026-11-06 15:05', 74800, '5?')]
waves(ax, med, MED, 2, 9, [1, 0, 1, 0, 1, 0])
ax.plot([T('2026-10-15 13:00'), T('2026-11-04 03:00')], [79800, 75690], color=MED, lw=0.6, alpha=0.6)
ax.plot([T('2026-10-21 16:00'), T('2026-11-04 15:15')], [84898, 81200], color=MED, lw=0.6, alpha=0.6)
ax.plot([T('2026-11-06 15:05'), T('2026-11-08 06:00'), T('2026-11-12'), T('2026-11-22')], [74800, 74300, 78750, 69100], color=BEAR, lw=1.5, ls='--')
ax.plot([T('2026-11-06 15:05'), T('2026-11-09'), T('2026-11-25')], [74800, 72600, 88500], color=BULL, lw=1.5, ls='--')
ax.text(T('2026-11-22'), 69100, ' main: bounce to 78.75k, then down', color=INK, fontsize=8.5, va='center')
ax.text(T('2026-11-25'), 88500, ' alt: wave 2 ends ~72.6k', color=INK, fontsize=8.5, va='center')
ax.set_xlim(T('2026-09-12'), T('2026-12-12')); ax.set_ylim(67500, 90000)
for y, t, c in [(84898, '84.9k  above = main count weakens', BULL), (72600, '72.6k  0.5 retrace & C=1.618A', SMALL),
                (69106, '69.1k  0.618 retrace', BEAR)]:
    level(ax, y, t, c, T('2026-09-13'))
style(ax, 'MEDIUM degree (4h)  ·  from 87.4k: 5 waves with overlap (wave 4 > wave 1 low) = diagonal')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# ---------- 3. SMALL degree (15m) ----------
ax = fig.add_subplot(gs[2]); d = M5[M5.index >= '2026-11-04 12:00'].resample('15min').agg(agg).dropna()
candles(ax, d, 0.007)
sm = [('2026-11-04 15:15', 81200, '4 (med)'), ('2026-11-04 19:55', 79903, 'i'), ('2026-11-04 22:45', 80941, 'ii'),
      ('2026-11-05 15:50', 75511, 'iii'), ('2026-11-05 16:20', 76980, 'a'), ('2026-11-05 23:10', 75144, 'b'),
      ('2026-11-06 00:40', 76400, 'c'), ('2026-11-06 07:30', 75253, 'd'), ('2026-11-06 14:50', 76116, 'e = iv'),
      ('2026-11-06 15:05', 74800, 'v?')]
waves(ax, sm, SMALL, 1.6, 8.5, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
ax.plot([T('2026-11-05 16:20'), T('2026-11-06 14:50')], [76980, 76116], color=SMALL, lw=0.7, alpha=0.7)
ax.plot([T('2026-11-05 23:10'), T('2026-11-06 14:35')], [75144, 75257], color=SMALL, lw=0.7, alpha=0.7)
ax.plot([T('2026-11-06 20:00'), T('2026-11-07 06:00'), T('2026-11-07 20:00')], [75920, 74300, 78000], color=BEAR, lw=1.5, ls='--')
ax.text(T('2026-11-07 20:00'), 78000, ' bounce targets 77.25k / 78.0k / 78.75k', color=INK, fontsize=8.5, va='center')
ax.set_xlim(T('2026-11-04 12:00'), T('2026-11-08 18:00')); ax.set_ylim(73500, 81800)
for y, t in [(74300, '74.3k  triangle thrust target'), (77245, '77.25k  0.382'), (78755, '78.75k  0.618')]:
    level(ax, y, t, SMALL, T('2026-11-04 12:10'))
style(ax, 'SMALL degree (15m)  ·  from 81.2k: i-ii-iii, triangle iv (a-e), v near 74.8k')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))

handles = [plt.Line2D([], [], color=c, lw=2, ls=ls) for c, ls in [(BIG, '-'), (MED, '-'), (SMALL, '-'), (BEAR, '--'), (BULL, '--')]]
fig.legend(handles, ['big degree', 'medium degree', 'small degree', 'main scenario (bear)', 'alternate (bull)'],
           loc='upper center', ncol=5, frameon=False, labelcolor=INK, fontsize=9, bbox_to_anchor=(0.5, 0.905))
fig.text(0.01, 0.003, 'BTCUSDT · Binance spot 1h to 2026-10-31 + OKX perp 5m to 2026-11-06 20:00 UTC · times UTC · '
         'counts are scenarios, not forecasts (auto wave-count direction accuracy here was 47–56%)', color=INK2, fontsize=8)
fig.savefig('logs/ew_count_now.png', dpi=110, facecolor=SURF, bbox_inches='tight')
print('saved')
