import os
import datetime
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app, origins=["https://crypto-prediction-api.netlify.app"])

model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route("/predict", methods=["GET"])
def predict():
    timeframe = request.args.get("timeframe", "1h")
    sample = pd.DataFrame([{ "open": 30000, "high": 30500, "low": 29900,
                             "close": 30300, "volume": 1000, "returns": 0, "volatility": 0 }])
    prediction = model.predict(sample)[0]
    prompt = (f"You are a top-tier crypto trading assistant. "
              f"Predicted BTC price: {prediction:.2f} USD. "
              f"Timeframe: {timeframe}. "
              f"Advise whether to LONG or SHORT, with Take-Profit and Stop-Loss levels.")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt}]}
    )
    ans = resp.json()["choices"][0]["message"]["content"]
    return jsonify({"prediction": f"{prediction:.2f}", "analysis": ans})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))