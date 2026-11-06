"""Download Binance spot klines from data.binance.vision and merge to one CSV.
usage: python fetch_data.py BTCUSDT 1h"""
import sys, io, zipfile, urllib.request, pandas as pd
sym, iv = (sys.argv + ['BTCUSDT', '1h'])[1:3]
base = 'https://data.binance.vision/data/spot/monthly/klines'
fr = []
for y in range(2017, 2031):
    for m in range(1, 13):
        u = f'{base}/{sym}/{iv}/{sym}-{iv}-{y}-{m:02d}.zip'
        try: b = urllib.request.urlopen(u, timeout=60).read()
        except Exception: continue
        z = zipfile.ZipFile(io.BytesIO(b))
        fr.append(pd.read_csv(z.open(z.namelist()[0]), header=None).iloc[:, :6])
df = pd.concat(fr); df.columns = ['t', 'o', 'h', 'l', 'c', 'v']
df['t'] = df['t'].where(df['t'] < 1e14, df['t'] // 1000)        # 2025+ files use microseconds
step = pd.Timedelta(iv).value // 10**6
df = df[df.t % step == 0].drop_duplicates('t').sort_values('t')  # drop malformed candles
df['time'] = pd.to_datetime(df.t, unit='ms', utc=True)
df[['time', 'o', 'h', 'l', 'c', 'v']].to_csv(f'data/{sym.lower()}_{iv}.csv', index=False)
print(len(df), df.time.iloc[0], df.time.iloc[-1])
