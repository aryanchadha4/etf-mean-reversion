import numpy as np
import pandas as pd

def compute_strategy_returns(prices, signal):
    positions = signal.shift(1)
    returns = prices.pct_change()
    strategy_returns = positions * returns
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    return returns, strategy_returns, cumulative_returns, positions

def calculate_metrics(strategy_returns):
    import numpy as np

    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    drawdown_series = cumulative_returns / cumulative_returns.cummax() - 1

    max_drawdown = drawdown_series.min(skipna=True)
    if isinstance(max_drawdown, pd.Series):
        max_drawdown = max_drawdown.values[0]  # handle edge case
    else:
        max_drawdown = float(max_drawdown)

    # If it's a DataFrame, select the first column
    if isinstance(strategy_returns, pd.DataFrame):
        strategy_returns = strategy_returns.iloc[:, 0]

    hit_rate = (strategy_returns > 0).mean()


    return sharpe, max_drawdown, hit_rate






def compute_quantitativo_returns(prices, signal):
    close = prices['Close']
    high = prices['High']
    returns = close.pct_change().fillna(0)
    sma_300 = close.rolling(window=300).mean()

    position = 0
    strategy_returns = pd.Series(0.0, index=close.index)

    for i in range(1, len(prices)):
        if position == 1:
            # Exit condition: today's close > yesterday's high
            if (close.iloc[i] < sma_300.iloc[i]):
                position = 0
            else:
                strategy_returns.iloc[i] = returns.iloc[i]

        # Entry condition: go long
        if position == 0 and signal.iloc[i] == 1:
            position = 1

    cumulative_returns = (1 + strategy_returns).cumprod()
    print("Number of trades:", signal.sum())
    print("Avg return per trade:", strategy_returns[strategy_returns != 0].mean())
    return returns, strategy_returns, cumulative_returns, strategy_returns != 0



def compute_quantitativo_longshort_returns(prices, signal):
    close = prices["Close"]
    high = prices["High"]
    low = prices["Low"]
    returns = close.pct_change().fillna(0)
    sma_300 = close.rolling(window=300).mean()

    # Volatility measure (20-day rolling std of returns)
    vol = close.pct_change().rolling(20).std()
    vol = vol.replace(0, np.nan).fillna(method='bfill')  # avoid division by zero

    base_risk = 0.02  # you can tune this

    strategy_returns = pd.Series(0.0, index=close.index)
    position = 0
    position_scaling = 0

    for i in range(1, len(prices)):
        # Exit conditions
        if position != 0:
            # Dynamic stop-loss exit
            if (position == 1 and close.iloc[i] < sma_300.iloc[i]) or \
               (position == -1 and close.iloc[i] > sma_300.iloc[i]):
                position = 0
                position_scaling = 0
            else:
                strategy_returns.iloc[i] = returns.iloc[i] * position_scaling * position

        # Entry conditions
        if position == 0:
            if signal.iloc[i] != 0:
                position = signal.iloc[i]
                position_scaling = base_risk / vol.iloc[i]

    cumulative_returns = (1 + strategy_returns).cumprod()

    print("Number of trades:", (signal != 0).sum())
    print("Avg return per trade:", strategy_returns[strategy_returns != 0].mean())

    return returns, strategy_returns, cumulative_returns, strategy_returns != 0

