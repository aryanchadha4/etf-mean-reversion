from strategy import generate_zscore_signal
from backtest import compute_strategy_returns, calculate_metrics
from utils import download_prices, plot_results
import pandas as pd
from strategy import generate_optimized_signal
from backtest import compute_optimized_returns
from strategy import generate_optimized_longshort_signal
from backtest import compute_optimized_longshort_returns

from strategy import train_filter_model, generate_optimized_ml_signal




# Parameters
TICKERS = ["SPY", "QQQ"]
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
WINDOW = 20
ENTRY_Z = 2.0
EXIT_Z = 0.5

trained_model = train_filter_model(download_prices("QQQ", START_DATE, END_DATE))

strategies = {
    "Z-Score": {
        "signal_func": generate_zscore_signal,
        "backtest_func": compute_strategy_returns
    },
    "Optimized": {
        "signal_func": generate_optimized_signal,
        "backtest_func": compute_optimized_returns
    },
    "Optimized Long-Short": {
        "signal_func": generate_optimized_longshort_signal,
        "backtest_func": compute_optimized_longshort_returns
    }, 
    "Optimized ML-Filtered": {
    "signal_func": lambda prices: generate_optimized_ml_signal(prices, trained_model),
    "backtest_func": compute_optimized_returns
    }

}

results = []
plot_data = []

for strategy_name, funcs in strategies.items():
    for ticker in TICKERS:
        print(f"\nBacktesting {ticker} with {strategy_name} strategy...")
        prices = download_prices(ticker, START_DATE, END_DATE)

        if strategy_name == "Z-Score":
            close = prices["Close"]
            signal, z_score = funcs["signal_func"](close, window=WINDOW, entry_z=ENTRY_Z, exit_z=EXIT_Z)
            returns, strategy_returns, cumulative_returns, positions = funcs["backtest_func"](close, signal)

        else:
            signal = funcs["signal_func"](prices)
            returns, strategy_returns, cumulative_returns, positions = funcs["backtest_func"](prices, signal)

        num_trades = (signal.diff().fillna(0) != 0).sum()
        avg_trade_return = strategy_returns[strategy_returns != 0].mean()

        sharpe, drawdown, hit_rate = calculate_metrics(strategy_returns)

        print(f"DEBUG: {strategy_name} | {ticker} | Sharpe: {sharpe} | Drawdown: {drawdown} | Hit Rate: {hit_rate}")
        print(f"Number of trades: {num_trades} | Avg return per trade: {avg_trade_return:.6f}")

        results.append({
            "Strategy": strategy_name,
            "Ticker": ticker,
            "Sharpe Ratio": round(sharpe, 3),
            "Max Drawdown": f"{drawdown:.2%}",
            "Hit Rate": f"{hit_rate:.2%}",
            "Num Trades": int(num_trades),
            "Avg Trade Return": f"{avg_trade_return:.4%}"
        })


        plot_data.append({
            "Strategy": strategy_name,
            "Ticker": ticker,
            "Dates": prices.index,
            "Cumulative Returns": cumulative_returns,
            "Returns": returns
        })

        action_log = pd.DataFrame(index=signal.index)
        action_log["Signal"] = signal

        action_log["Action"] = signal.map({1: "Buy/Long", -1: "Sell/Short", 0: "Hold"})

        action_log["Strategy"] = strategy_name
        action_log["Ticker"] = ticker

        # Filter only dates where action is not hold
        action_log = action_log[action_log["Action"] != "Hold"]

        if "all_actions" not in locals():
            all_actions = []
        all_actions.append(action_log)



import matplotlib.pyplot as plt


for res in plot_data:
    plt.figure(figsize=(12, 5))
    plt.plot(res["Dates"], res["Cumulative Returns"], label=f"{res['Strategy']} Strategy")
    plt.plot(res["Dates"], (1 + res["Returns"].fillna(0)).cumprod(), label="Buy & Hold")
    plt.title(f"{res['Ticker']} - {res['Strategy']} Mean Reversion")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from xgboost import plot_importance

plot_importance(trained_model)
plt.title("XGBoost Feature Importance")
plt.show()


# Save results
print("Saving results to CSV...")
df = pd.DataFrame(results)
print(df)
df.to_csv("outputs/multi_etf_results.csv", index=False)


# Combine all action logs into one DataFrame
all_actions_df = pd.concat(all_actions)
all_actions_df.to_csv("outputs/trade_actions_log.csv")