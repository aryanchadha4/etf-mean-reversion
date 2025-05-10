import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def download_prices(ticker, start, end):
    import yfinance as yf
    data = yf.download(ticker, start=start, end=end)

    # Fix any multi-index issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)  # Flatten the columns

    return data[["High", "Low", "Close"]]

def plot_results(dates, strategy_cumulative, benchmark_returns):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, strategy_cumulative, label="Strategy")
    plt.plot(dates, (1 + benchmark_returns.fillna(0)).cumprod(), label="Buy & Hold")
    plt.legend()
    plt.title("Mean Reversion Strategy vs Buy & Hold")
    plt.grid(True)
    plt.show()

import os

def log_trades(prices, positions, output_path="outputs/trades.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Identify entry/exit points
    trades = positions.diff().fillna(0) != 0

    # Build trade log
    trade_log = pd.DataFrame({
        "Date": prices.index[trades],
        "Price": prices[trades],
        "Position": positions[trades]
    })

    trade_log.to_csv(output_path, index=False)

