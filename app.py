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
    prompt = (f"You are a top-tier crypto trading assistant. "
              f"Predicted BTC price: {prediction:.2f} USD. "
              f"Timeframe: {tf_label}. "
              f"Advise whether to LONG or SHORT, with Take-Profit and Stop-Loss levels.")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}]}
    )
    result = resp.json()
    print("Groq API raw response:", result)  # Debug log to Render logs
    if "choices" not in result:
        return jsonify({"error": result}), 500
    ans = result["choices"][0]["message"]["content"]
    import re
    # Try to extract predicted price from the LLM response
    match = re.search(r'price of \$?([0-9,\.]+)', ans)
    predicted_price_from_llm = float(match.group(1).replace(',', '')) if match else None

    # Compute % change based on last known BTC price
    current_price = df["close"].iloc[-1]
    if predicted_price_from_llm:
        change_pct = ((predicted_price_from_llm - current_price) / current_price) * 100
    else:
        change_pct = None

    return jsonify({
        "prediction": f"{prediction:.2f}",
        "analysis": ans,
        "price": predicted_price_from_llm,
        "change": change_pct
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))