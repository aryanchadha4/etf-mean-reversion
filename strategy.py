import pandas as pd

def generate_zscore_signal(prices, window=20, entry_z=1.5, exit_z=0.5):
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    z_score = (prices - rolling_mean) / rolling_std

    signal = pd.Series(0.0, index=prices.index)
    signal[z_score < -entry_z] = 1     # Long
    signal[z_score > entry_z] = -1     # Short
    signal[(z_score > -exit_z) & (z_score < exit_z)] = 0  # Exit

    return signal, z_score