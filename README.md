# Elliott wave bot (BTCUSDT)

Research code. Nothing here places real orders.

- `engine.py` — causal zigzag (pivots carry the bar they became known), wave rules.
- `bt3.py` — current strategy: wave 1 at big degree, entry on the first lower-degree
  swing inside the wave-2 retrace zone, stop just beyond that swing (max 1–2%).
- `fetch_data.py` — rebuild `data/*.csv` from Binance public archives.

Backtests split 2017–2022 (choose settings) / 2023–now (unseen check), risk 1%/trade,
fees 0.1%/side + slippage.
