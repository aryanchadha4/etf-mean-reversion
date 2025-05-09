from strategy import generate_zscore_signal
from backtest import compute_strategy_returns, calculate_metrics
from utils import download_prices, plot_results
import pandas as pd

# Parameters
TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
WINDOW = 20
ENTRY_Z = 2.0
EXIT_Z = 0.5

results = []
plot_data = []  # For charting only

for ticker in TICKERS:
    print(f"\nBacktesting {ticker}...")
    prices = download_prices(ticker, START_DATE, END_DATE)

    signal, _ = generate_zscore_signal(prices, window=WINDOW, entry_z=ENTRY_Z, exit_z=EXIT_Z)
    returns, strategy_returns, cumulative_returns, positions = compute_strategy_returns(prices, signal)
    sharpe, drawdown, hit_rate = calculate_metrics(strategy_returns)

    # Save summary metrics (for CSV)
    results.append({
        "Ticker": ticker,
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown": f"{drawdown:.2%}",
        "Hit Rate": f"{hit_rate:.2%}"
    })

    # Save time series for plotting
    plot_data.append({
        "Ticker": ticker,
        "Dates": prices.index,
        "Cumulative Returns": cumulative_returns,
        "Returns": returns
    })



import matplotlib.pyplot as plt


for res in plot_data:
    plt.figure(figsize=(12, 5))
    plt.plot(res["Dates"], res["Cumulative Returns"], label="Strategy")
    plt.plot(res["Dates"], (1 + res["Returns"].fillna(0)).cumprod(), label="Buy & Hold")
    plt.title(f"{res['Ticker']} Mean Reversion Strategy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Save results
df = pd.DataFrame(results)
df.to_csv("outputs/multi_etf_results.csv", index=False)


