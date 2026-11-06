"""Download OKX candles (default BTC-USDT-SWAP 5m, full history) -> data/okx_<inst>_<bar>.csv
usage: python retr/fetch_okx.py BTC-USDT-SWAP 5m"""
import sys, time, json, subprocess, pandas as pd
inst, bar = (sys.argv + ['BTC-USDT-SWAP', '5m'])[1:3]
url = 'https://www.okx.com/api/v5/market/history-candles?instId={}&bar={}&limit=100{}'
rows, after = [], ''
while True:
    for k in range(5):
        try:
            out = subprocess.run(['curl', '-s', '-m', '20', url.format(inst, bar, after)], capture_output=True, text=True).stdout
            d = json.loads(out)['data']; break
        except Exception as e:
            time.sleep(2 ** k)
    else:
        raise SystemExit('network failed')
    if not d: break
    rows += d
    after = f'&after={d[-1][0]}'
    if len(rows) % 20000 == 0: print(len(rows), pd.to_datetime(int(d[-1][0]), unit='ms'), flush=True)
    time.sleep(0.11)                                   # OKX limit: 20 req / 2 s
df = pd.DataFrame(rows).iloc[:, :8]
df.columns = ['t', 'o', 'h', 'l', 'c', 'vol', 'volccy', 'v']   # v = volume in quote (USDT)
df = df[pd.DataFrame(rows)[8] == '1'] if len(rows[0]) > 8 else df   # confirmed candles only
df['time'] = pd.to_datetime(df.t.astype('int64'), unit='ms', utc=True)
df = df.drop_duplicates('time').sort_values('time')
for k in 'ohlcv': df[k] = df[k].astype(float)
out = f"data/okx_{inst.lower()}_{bar}.csv"
df[['time', 'o', 'h', 'l', 'c', 'v']].to_csv(out, index=False)
print(out, len(df), df.time.iloc[0], df.time.iloc[-1])
