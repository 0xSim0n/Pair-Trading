import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def pair_trading_signals(ticker1, ticker2, start='2021-01-01', end='2024-12-31', window=30, entry_threshold=2, exit_threshold=0):
    data1 = yf.download(ticker1, start=start, end=end)['Close']
    data2 = yf.download(ticker2, start=start, end=end)['Close']

    prices = pd.DataFrame({
        ticker1: data1,
        ticker2: data2
    }).dropna()

    X = sm.add_constant(prices[ticker1])
    model = sm.OLS(prices[ticker2], X).fit()
    beta = model.params[ticker1]
    spread = prices[ticker2] - beta * prices[ticker1]

    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std

    signals = pd.DataFrame(index=prices.index)
    signals['spread'] = spread
    signals['z_score'] = z_score
    signals['position'] = 0
    signals.loc[z_score > entry_threshold, 'position'] = -1  # short spread
    signals.loc[z_score < -entry_threshold, 'position'] = 1  # long spread
    signals['position'] = signals['position'].replace(0, method='ffill').fillna(0)

    plt.figure(figsize=(15, 7))
    plt.plot(signals.index, signals['z_score'], label='Z-score', color='blue')
    plt.axhline(entry_threshold, color='red', linestyle='--', label='Entry threshold')
    plt.axhline(-entry_threshold, color='green', linestyle='--')
    plt.axhline(0, color='black', linestyle=':')
    plt.title(f'Z-score Spread {ticker1}/{ticker2}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return signals

signals = pair_trading_signals(ticker1='ETH', ticker2='LINK')
