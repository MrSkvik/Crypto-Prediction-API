import os
import datetime
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import re

load_dotenv()
app = Flask(__name__)
CORS(app, origins=["https://crypto-prediction-api.netlify.app"])

model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    timeframe = data.get("timeframe", "1h")
    # Map frontend timeframe to model/feature preparation
    interval_map = {
        "20m": "20 minutes",
        "1h": "1 hour",
        "24h": "24 hours"
    }
    tf_label = interval_map.get(timeframe, timeframe)

    # Fetch recent BTC/USDT data from Binance API based on timeframe
    binance_interval_map = {
        "20m": "1m",   # Approximate 20 minutes using last 20 1-minute candles
        "1h": "1m",    # Approximate 1 hour using last 60 1-minute candles
        "24h": "1h"    # Approximate 24 hours using last 24 1-hour candles
    }
    interval = binance_interval_map.get(timeframe, "1m")
    limit_map = {"20m": 20, "1h": 60, "24h": 24}
    limit = limit_map.get(timeframe, 20)

    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
            timeout=10
        )
        resp.raise_for_status()
        klines = resp.json()
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df = df.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float
        })
        # Simple feature engineering
        returns = (df["close"].iloc[-1] - df["open"].iloc[0]) / df["open"].iloc[0]
        volatility = ((df["high"] - df["low"]) / df["high"]).std()
        sample = pd.DataFrame([{
            "open": df["open"].iloc[-1],
            "high": df["high"].iloc[-1],
            "low": df["low"].iloc[-1],
            "close": df["close"].iloc[-1],
            "volume": df["volume"].iloc[-1],
            "returns": returns,
            "volatility": volatility
        }])
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {str(e)}"}), 500

    prediction = model.predict(sample)[0]
    prompt = (
        "You are an expert crypto analyst. Based on the following data, give a clear, short forecast for Bitcoin (BTC). "
        "Include whether to LONG or SHORT, with exact Take Profit (TP) and Stop Loss (SL) prices. Keep it simple.\n\n"
        f"XGBoost Prediction: ${prediction:.2f}\n"
        f"- Current Price: ${df['close'].iloc[-1]:,.2f}\n"
        f"- Volume: ${df['volume'].iloc[-1]:,.2f}M\n"
        f"- Timeframe: {tf_label}\n"
        f"- Volatility: medium\n"
        f"- Momentum: bullish\n\n"
        "Format:\n"
        "Prediction: [price]\n"
        "Signal: [Long/Short]\n"
        "Take Profit: [price]\n"
        "Stop Loss: [price]\n"
        "Explanation: [max 1 sentence]"
    )
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}]}
    )
    result = resp.json()
    print("Groq API raw response:", result)

    if isinstance(result, dict) and "choices" in result and result["choices"]:
        ans = result["choices"][0]["message"]["content"].strip()
    else:
        signal = "Long" if float(prediction) > float(df["close"].iloc[-1]) else "Short"
        tp_fallback = round(float(prediction), 2)
        sl_fallback = round(float(df["close"].iloc[-1]) * (0.994 if signal == "Long" else 1.006), 2)
        ans = (
            f"Prediction: ${float(prediction):.2f}\n"
            f"Signal: {signal}\n"
            f"Take Profit: ${tp_fallback}\n"
            f"Stop Loss: ${sl_fallback}\n"
            f"Explanation: XGBoost target vs current ${float(df['close'].iloc[-1]):.2f}."
        )

    current_price = float(df["close"].iloc[-1])
    pred_num = float(prediction)
    change_pct = ((pred_num - current_price) / current_price) * 100.0 if current_price > 0 else 0.0
    signal_final = "Long" if pred_num > current_price else "Short"
    tp_numeric = round(pred_num, 2)
    sl_numeric = round(current_price * (0.994 if signal_final == "Long" else 1.006), 2)

    return jsonify({
        "prediction": round(pred_num, 2),
        "analysis": ans,
        "price": round(pred_num, 2),
        "change": round(change_pct, 2),
        "tp": tp_numeric,
        "sl": sl_numeric
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))