import ccxt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

exchange = ccxt.binance()
symbols = [s['symbol'] for s in exchange.load_markets().values() if '/USDT' in s['symbol'] and s['active']]

data = []

for symbol in symbols[:50]:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '5m')[-3:]
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        pct_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        vol_change = df['volume'].iloc[-1] - df['volume'].iloc[0]
        data.append([symbol, pct_change, vol_change])
    except:
        continue

df = pd.DataFrame(data, columns=['symbol', 'pct_change', 'vol_change'])

# Кластеризация
X = StandardScaler().fit_transform(df[['pct_change', 'vol_change']])
clustering = DBSCAN(eps=0.8, min_samples=3).fit(X)
df['cluster'] = clustering.labels_

# Выводим аномалии
anomalies = df[df['cluster'] == -1]
print("📊 Обнаружены аномалии:\n", anomalies)