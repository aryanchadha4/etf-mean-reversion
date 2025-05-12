import pandas as pd

def generate_zscore_signal(prices, window=20, entry_z=2.0, exit_z=0.5):
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    z_score = (prices - rolling_mean) / rolling_std

    signal = pd.Series(0, index=prices.index)
    signal[z_score < -entry_z] = 1     # Long
    signal[z_score > entry_z] = -1     # Short
    signal[(z_score > -exit_z) & (z_score < exit_z)] = 0  # Exit

    return signal, z_score


def generate_optimized_signal(prices):
    high = prices['High']
    low = prices['Low']
    close = prices['Close']

    rolling_hl = (high - low).rolling(window=25).mean()
    rolling_high = high.rolling(window=10).max()
    lower_band = rolling_high - 2.2 * rolling_hl
    IBS = (close - low) / (high - low)

    # Entry signal: 1 when entry condition met, else 0
    signal = ((close < lower_band) & (IBS < 0.3)).astype(int)
    return signal

def generate_optimized_longshort_signal(prices):
    high = prices['High']
    low = prices['Low']
    close = prices['Close']

    hl_mean = (high - low).rolling(window=25).mean()
    rolling_high = high.rolling(window=10).max()
    rolling_low = low.rolling(window=10).min()
    lower_band = rolling_high - 2.2 * hl_mean
    upper_band = rolling_low + 2.2 * hl_mean
    IBS = (close - low) / (high - low)
    sma_300 = close.rolling(window=300).mean()

    signal = pd.Series(0, index=prices.index)

    # Long condition
    long_condition = (close < lower_band) & (IBS < 0.3) & (close > sma_300)
    signal[long_condition] = 1

    # Short condition
    short_condition = (close > upper_band) & (IBS > 0.7) & (close < sma_300)
    signal[short_condition] = -1

    return signal

def generate_optimized_ml_signal(prices, model):
    high = prices['High']
    low = prices['Low']
    close = prices['Close']

    rolling_hl = (high - low).rolling(window=25).mean()
    rolling_high = high.rolling(10).max()
    lower_band = rolling_high - 2.2 * rolling_hl
    IBS = (close - low) / (high - low)

    entry_condition = (close < lower_band) & (IBS < 0.3)

    features = pd.DataFrame({
        'IBS': IBS,
        'BandDiff': rolling_high - 2.2 * rolling_hl
    }).fillna(0)

    ml_filter = model.predict(features)
    signal = (entry_condition & (ml_filter == 1)).astype(int)
    return signal




from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def generate_features(prices):
    high = prices['High']
    low = prices['Low']
    close = prices['Close']

    df = pd.DataFrame(index=prices.index)
    df['IBS'] = (close - low) / (high - low)
    df['BandDiff'] = high.rolling(10).max() - 2.2 * (high - low).rolling(25).mean()
    df['Return+1'] = close.pct_change().shift(-1)
    df['Target'] = (df['Return+1'] > 0).astype(int)
    df = df.dropna()
    return df

def train_filter_model(prices):
    df = generate_features(prices)
    X = df[['IBS', 'BandDiff']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model
