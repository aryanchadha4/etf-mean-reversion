import numpy as np
import pandas as pd
from strategy import generate_zscore_signal
from backtest import compute_strategy_returns, calculate_metrics
from utils import download_prices

# Settings
TICKER = "SPY"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
WINDOW = 20

entry_z_values = np.arange(1.0, 2.1, 0.25)
exit_z_values = np.arange(0.25, 1.1, 0.25)

results = []

# Load data once
prices = download_prices(TICKER, START_DATE, END_DATE)

# Grid search
for entry_z in entry_z_values:
    for exit_z in exit_z_values:
        signal, _ = generate_zscore_signal(prices, window=WINDOW, entry_z=entry_z, exit_z=exit_z)
        returns, strategy_returns, _, _ = compute_strategy_returns(prices, signal)
        sharpe, drawdown, hit_rate = calculate_metrics(strategy_returns)
        
        results.append({
            "Entry_Z": entry_z,
            "Exit_Z": exit_z,
            "Sharpe": sharpe,
            "Max Drawdown": drawdown,
            "Hit Rate": hit_rate
        })

# Convert to DataFrame
df = pd.DataFrame(results)
df.sort_values(by="Sharpe", ascending=False, inplace=True)

# Output results
print(df.head(10))  # Top 10 parameter sets
df.to_csv("outputs/grid_search_results.csv", index=False)
