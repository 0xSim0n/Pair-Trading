import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def pair_trading_signals(ticker1, ticker2, start='2021-07-01', end='2024-06-30',
                         window=30, entry_threshold=1, exit_threshold=0):

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
    signals.loc[zscore.abs() < exit_threshold, 'position'] = 0
    signals['position'] = signals['position'].ffill().fillna(0)

    returns1 = np.log(df[ticker1] / df[ticker1].shift(1))
    returns2 = np.log(df[ticker2] / df[ticker2].shift(1))
    position = signals['position'].shift(1)

    pnl = position * (returns1 - beta * returns2)
    cumulative = pnl.cumsum()

    sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(365)

    entry_points = (signals['position'].diff() != 0) & (signals['position'] != 0)
    exit_points = (signals['position'].diff() != 0) & (signals['position'] == 0)

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(cumulative, label='Equity Curve', color='purple')
    axs[0].set_title(f'Pair Trading Backtest: {ticker1} vs {ticker2}\nSharpe Ratio: {sharpe_ratio:.2f}')
    axs[0].set_ylabel('Cumulative Return')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(signals.index, signals['zscore'], label='Z-score', color='blue')
    axs[1].axhline(entry_threshold, color='red', linestyle='--', label='Entry threshold')
    axs[1].axhline(-entry_threshold, color='green', linestyle='--')
    axs[1].axhline(0, color='black', linestyle=':')
    axs[1].plot(signals.index[entry_points], signals['zscore'][entry_points], 'rx', label='Entry', markersize=8)
    axs[1].plot(signals.index[exit_points], signals['zscore'][exit_points], 'kx', label='Exit', markersize=8)
    axs[1].set_title(f'Z-score spread: {ticker1} vs {ticker2}')
    axs[1].set_ylabel('Z-score')
    axs[1].grid(True)
    axs[1].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

    return signals, pnl, cumulative, sharpe_ratio


signals, daily_pnl, cumulative, sharpe = pair_trading_signals('BTC-USD', 'ETH-USD')
print(f"Sharpe Ratio: {sharpe:.2f}")
