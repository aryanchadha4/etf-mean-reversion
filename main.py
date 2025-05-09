from strategy import generate_zscore_signal
from backtest import compute_strategy_returns, calculate_metrics
from utils import plot_results, download_prices
from utils import log_trades


# Parameters
TICKER = "SPY"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
WINDOW = 20
ENTRY_Z = 1.5
EXIT_Z = 0.5

# Load data
prices = download_prices(TICKER, START_DATE, END_DATE)

# Run strategy
signal, z_score = generate_zscore_signal(prices, WINDOW, ENTRY_Z, EXIT_Z)

# Backtest
returns, strategy_returns, cumulative_returns, positions = compute_strategy_returns(prices, signal)

sharpe, max_drawdown, hit_rate = calculate_metrics(strategy_returns)

# Output results
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Hit Rate: {hit_rate:.2%}")

# Save trade log
log_trades(prices, positions)

# Plot
plot_results(prices.index, cumulative_returns, returns)

