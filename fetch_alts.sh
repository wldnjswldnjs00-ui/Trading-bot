#!/bin/bash
# Binance USDT-M futures 5m klines, 2023-11..2026-10, for the coins listed.
cd "$(dirname "$0")/data/alt5m"
for s in BTC ETH SOL XRP DOGE BNB ADA AVAX LINK DOT LTC BCH TRX NEAR APT ARB OP SUI INJ TIA SEI WIF 1000PEPE FET ATOM FIL ETC UNI AAVE ORDI; do
  for ym in $(python3 -c "import pandas as pd;print(' '.join(pd.period_range('2023-11','2026-10',freq='M').strftime('%Y-%m')))"); do
    f=${s}USDT-5m-$ym.zip; [ -f $f ] || curl -sSf -o $f https://data.binance.vision/data/futures/um/monthly/klines/${s}USDT/5m/$f 2>/dev/null || rm -f $f
  done; echo "$s $(ls ${s}USDT-5m-* 2>/dev/null | wc -l)"
done
