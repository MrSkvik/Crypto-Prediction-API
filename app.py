import os
import datetime
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import re
import math

load_dotenv()
app = Flask(__name__)
CORS(app, origins=["https://crypto-prediction-api.netlify.app"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    timeframe = data.get("timeframe", "1h")

    # Display labels for the UI
    interval_map = {"20m": "20 minutes", "1h": "1 hour", "24h": "24 hours"}
    tf_label = interval_map.get(timeframe, timeframe)

    # Sampling plan for Binance data
    binance_interval_map = {"20m": "1m", "1h": "1m", "24h": "1h"}
    limit_map = {"20m": 20, "1h": 60, "24h": 24}
    interval = binance_interval_map.get(timeframe, "1m")
    limit = limit_map.get(timeframe, 20)

    # --- Fetch recent candles from Binance ---
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        klines = r.json()
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ],
        ).astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {str(e)}"}), 500

    # --- Build features for the model ---
    try:
        # Return over the sampled window and a simple intrabar volatility proxy
        returns = (df["close"].iloc[-1] - df["open"].iloc[0]) / df["open"].iloc[0]
        volatility = ((df["high"] - df["low"]) / df["high"]).std()

        try:
            vol_val = float(volatility)
            if not math.isfinite(vol_val):
                vol_val = 0.01
        except Exception:
            vol_val = 0.01

        sample = pd.DataFrame([{
            "open": df["open"].iloc[-1],
            "high": df["high"].iloc[-1],
            "low": df["low"].iloc[-1],
            "close": df["close"].iloc[-1],
            "volume": df["volume"].iloc[-1],
            "returns": float(returns),
            "volatility": float(vol_val),
        }])
    except Exception as e:
        return jsonify({"error": f"Feature build failed: {str(e)}"}), 500

    current_price = float(df["close"].iloc[-1])

    # --- XGBoost: single source of truth for the target price ---
    try:
        feature_order = ["open", "high", "low", "close", "volume", "returns", "volatility"]
        dmat = xgb.DMatrix(sample[feature_order].to_numpy())
        xgb_pred = float(model.get_booster().predict(dmat)[0])
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    # --- Derive trading signal and TP/SL from the model prediction ---
    signal = "Long" if xgb_pred >= current_price else "Short"
    # Risk factor from volatility, clamped to 0.5%–2%
    risk = min(0.02, max(0.005, vol_val))
    tp = xgb_pred * (1 + risk) if signal == "Long" else xgb_pred * (1 - risk)
    sl = xgb_pred * (1 - risk) if signal == "Long" else xgb_pred * (1 + risk)

    # --- One‑sentence rationale via Groq (no numbers allowed) ---
    explanation = "Model momentum and liquidity favor this direction for the selected timeframe."
    try:
        if GROQ_API_KEY:
            vol_bucket = "low" if risk <= 0.0075 else ("medium" if risk <= 0.015 else "high")
            prompt = (
                "You are a professional crypto analyst.\n"
                f"Timeframe: {tf_label}\n"
                f"Direction: {signal}\n"
                f"Volatility: {vol_bucket}\n\n"
                "Write ONE short sentence (max 18 words) explaining the signal.\n"
                "Do NOT include any numbers or prices."
            )
            g = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=10,
            )
            gj = g.json()
            if isinstance(gj, dict) and gj.get("choices"):
                explanation = gj["choices"][0]["message"]["content"].strip()
    except Exception:
        # keep default explanation on any LLM error
        pass

    # --- Compose analysis text strictly from OUR numbers ---
    analysis_text = (
        f"Prediction: ${xgb_pred:,.2f}\n"
        f"Signal: {signal}\n"
        f"Take Profit: ${tp:,.2f}\n"
        f"Stop Loss: ${sl:,.2f}\n"
        f"Explanation: {explanation}"
    )

    change_pct = ((xgb_pred - current_price) / current_price) * 100.0 if current_price > 0 else 0.0

    return jsonify({
        "prediction": round(xgb_pred, 2),        # Authoritative model target
        "price": round(xgb_pred, 2),             # Kept for frontend compatibility
        "current_price": round(current_price, 2),
        "change": round(change_pct, 2),          # Computed from model vs. live price
        "signal": signal,
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "analysis": analysis_text,               # Human string built from our numbers
        "source": "xgboost"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))