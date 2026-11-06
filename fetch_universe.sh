#!/bin/bash
# 5m USDT-M futures klines 2023-11..2026-10 for every coin in universe150.txt (8 in parallel)
cd "$(dirname "$0")/data/alt5m"
months=$(python3 -c "import pandas as pd;print(' '.join(pd.period_range('2023-11','2026-10',freq='M').strftime('%Y-%m')))")
for s in $(cat ../../universe150.txt); do for m in $months; do echo "$s $m"; done; done | \
xargs -P 8 -n 2 sh -c 'f=${0}USDT-5m-$1.zip; [ -f $f ] || curl -sSf -o $f https://data.binance.vision/data/futures/um/monthly/klines/${0}USDT/5m/$f 2>/dev/null || rm -f $f'
ls | sed 's/USDT-5m.*//' | sort | uniq -c | wc -l
