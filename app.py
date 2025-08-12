import os
import requests
import xgboost as xgb
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone

DIR_PATH = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

app = Flask(__name__)
CORS(app, origins=["https://crypto-prediction-api.netlify.app"])  # Netlify frontend

MODEL_PATH = os.path.join(DIR_PATH, "xgb_model.json")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
HEADERS = {"User-Agent": "crypto-prediction-api/1.0"}

TF_MAP = {
    "20m": (60, 20),     # 20 x 1m
    "1h":  (300, 12),    # 12 x 5m
    "24h": (3600, 24),   # 24 x 1h
}

def fetch_coinbase_candles(granularity: int):
    r = requests.get(COINBASE_CANDLES, params={"granularity": granularity}, headers=HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json()
    # Coinbase returns [time, low, high, open, close, volume], newest first. Sort ascending.
    data.sort(key=lambda x: x[0])
    return data

def prepare_features(candles, lookback: int):
    """
    Build a single-row pandas DataFrame with the exact feature names the model
    was trained with: open, high, low, close, volume, returns, volatility.
    Coinbase candle format: [time, low, high, open, close, volume] (ascending).
    """
    window = candles[-lookback:] if len(candles) >= lookback else candles[:]
    if not window:
        raise ValueError("Not enough candles returned from Coinbase.")

    last = window[-1]
    # Coinbase entry: [time, low, high, open, close, volume]
    open_p = float(last[3])
    high_p = float(last[2])
    low_p = float(last[1])
    close_p = float(last[4])
    volume = float(last[5])

    # Simple features
    returns = (close_p - open_p) / open_p if open_p else 0.0
    highs = np.array([float(c[2]) for c in window], dtype=float)
    lows  = np.array([float(c[1]) for c in window], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        volatility = np.nan_to_num(((highs - lows) / np.maximum(highs, 1e-9)).std(), nan=0.0)

    features_df = pd.DataFrame([{
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": volume,
        "returns": returns,
        "volatility": float(volatility),
    }], columns=["open","high","low","close","volume","returns","volatility"])

    return features_df, close_p, float(volatility)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        tf = (payload.get("timeframe") or "1h").lower()
        gran, lookback = TF_MAP.get(tf, (300, 12))

        candles = fetch_coinbase_candles(gran)
        features_df, current_price, vol = prepare_features(candles, lookback)

        try:
            # Prefer scikit-learn interface with named columns
            pred = float(model.predict(features_df)[0])
        except Exception:
            # Fallback to native Booster if the wrapper is not compatible
            booster = model.get_booster()
            pred = float(booster.predict(xgb.DMatrix(features_df.values))[0])

        change_pct = (pred - current_price) / current_price * 100.0 if current_price else 0.0
        signal = "Long" if pred >= current_price else "Short"

        recent = candles[-lookback:]
        highs = np.array([c[2] for c in recent], dtype=float)
        lows  = np.array([c[1] for c in recent], dtype=float)
        rng = float(np.nan_to_num((highs - lows).mean(), nan=0.0))
        cushion = max(0.002, min(0.02, rng / max(current_price, 1e-9)))  # 0.2%..2%

        if signal == "Long":
            tp = pred * (1 + cushion)
            sl = current_price * (1 - cushion)
        else:
            tp = pred * (1 - cushion)
            sl = current_price * (1 + cushion)

        analysis = f"XGBoost on Coinbase BTC-USD ({lookback} bars @ {gran}s)."

        return jsonify({
            "prediction": round(pred, 2),
            "current_price": round(current_price, 2),
            "change": round(change_pct, 2),
            "signal": signal,
            "tp": round(float(tp), 2),
            "sl": round(float(sl), 2),
            "analysis": analysis,
            "time_utc": datetime.now(timezone.utc).isoformat(timespec='seconds')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))