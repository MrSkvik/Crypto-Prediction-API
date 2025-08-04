import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# Binance API endpoint
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Parameters
SYMBOL = "BTCUSDT"
INTERVAL = "1m"  # you can change to '1h', '4h', etc.
LIMIT = 1000

# Get candlestick data from Binance
def fetch_binance_data(symbol, interval, limit):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(BINANCE_URL, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["close"].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

# Fetch and prepare data
data = fetch_binance_data(SYMBOL, INTERVAL, LIMIT)

# Features and target
features = ["open", "high", "low", "close", "volume", "returns", "volatility"]
X = data[features]
y = data["close"].shift(-1).dropna()  # predict next close price
X = X.iloc[:-1]  # align with shifted target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Model trained. MSE: {mse:.4f}")

# Save model
model.save_model("xgb_model.json")
print("Model saved to xgb_model.json")