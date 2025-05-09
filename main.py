import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
TICKER = "SPY"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
WINDOW = 20
ENTRY_Z = 1.5

# Load Data
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
prices = data['Close'].copy()
if isinstance(prices, pd.DataFrame) and prices.shape[1] == 1:
    prices = prices.iloc[:, 0]


# Calculate Rolling Mean, Std, Z-Score
rolling_mean = prices.rolling(WINDOW).mean()
rolling_std = prices.rolling(WINDOW).std()
z_score = (prices - rolling_mean) / rolling_std

# Generate Trading Signals
signal = pd.Series(0.0, index=prices.index)
signal[z_score < -ENTRY_Z] = 1     # Long entry
signal[z_score > ENTRY_Z] = -1     # Short entry
signal[(z_score > -0.5) & (z_score < 0.5)] = 0  # Exit


# Calculate Strategy Returns
positions = signal.shift(1)  # Shift to avoid look-ahead bias
returns = prices.pct_change()
strategy_returns = positions * returns


# Calculate Performance Metrics
cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
hit_rate = (strategy_returns > 0).mean()

# Print Metrics
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Hit Rate: {hit_rate:.2%}")

# Plot Returns
plt.figure(figsize=(14,6))
plt.plot(cumulative_returns, label="Strategy")
plt.plot((1 + returns.fillna(0)).cumprod(), label="Buy & Hold")
plt.legend()
plt.title(f"{TICKER} Mean Reversion Strategy")
plt.grid(True)
plt.show()
