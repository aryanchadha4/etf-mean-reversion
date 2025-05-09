import numpy as np
import pandas as pd

def compute_strategy_returns(prices, signal):
    positions = signal.shift(1)
    returns = prices.pct_change()
    strategy_returns = positions * returns
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    return returns, strategy_returns, cumulative_returns, positions

def calculate_metrics(strategy_returns):
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    max_drawdown = (1 + strategy_returns.fillna(0)).cumprod().div(
        (1 + strategy_returns.fillna(0)).cumprod().cummax()).sub(1).min()
    hit_rate = (strategy_returns > 0).mean()
    return sharpe, max_drawdown, hit_rate
