import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def pair_trading_signals(ticker1, ticker2, start='2021-01-01', end='2024-12-31',
                         window=30, entry_threshold=2, exit_threshold=0):

    data1 = yf.download(ticker1, start=start, end=end)['Close']
    data2 = yf.download(ticker2, start=start, end=end)['Close']

    if data1.empty or data2.empty:
        raise ValueError("Data error")

    df = pd.concat([data1, data2], axis=1)
    df.columns = [ticker1, ticker2]
    df = df.dropna()

    X = sm.add_constant(df[ticker1])
    model = sm.OLS(df[ticker2], X).fit()
    beta = model.params[ticker1]
    spread = df[ticker2] - beta * df[ticker1]

    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    zscore = (spread - spread_mean) / spread_std

    signals = pd.DataFrame(index=df.index)
    signals['zscore'] = zscore
    signals['position'] = 0
    signals.loc[zscore > entry_threshold, 'position'] = -1
    signals.loc[zscore < -entry_threshold, 'position'] = 1
    signals['position'] = signals['position'].ffill().fillna(0)

    plt.figure(figsize=(14, 6))
    plt.plot(signals.index, signals['zscore'], label='Z-score', color='blue')
    plt.axhline(entry_threshold, color='red', linestyle='--', label='Entry threshold')
    plt.axhline(-entry_threshold, color='green', linestyle='--', label='Entry threshold')
    plt.axhline(0, color='black', linestyle=':')
    plt.title(f'Z-score spread: {ticker1} vs {ticker2}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return signals

signals = pair_trading_signals('ETH-USD', 'BTC-USD')
