# 🏆 Gold (XAU/USD) AI Swing Trader

An institutional-grade Machine Learning model that forecasts the direction of Gold prices over a 5-day horizon and recommennds whether to buy or not. Built with **XGBoost** and deployed via **Streamlit**, this tool focuses on macro-economic drivers rather than lagging technical indicators.

**Performance:** \~60% Precision on Buy Signals (Out-of-Sample 2024-Present).

---

## 🧠 The Strategy

Unlike retail strategies, this AI analyzes the Physics of the market. It uses 4 features to detect high-probability setups:

1.  **Real Yields:** Tracks the 5-day change in US 10Y Yields. When yields spike, Gold often drops.
2.  **Volatility Regime:** Uses **Parkinson Volatility** to detect volatility compression. Breakouts often follow low-volatility periods.
3.  **DXY Correlation:** Measures if Gold is moving inversely to the Dollar Index. The model avoids buying when this correlation breaks.
4.  **Market Structure:** Calculates Gold's position relative to its 20-Day High/Low range.

**Model Type:** Long-Only (Buy vs. Cash). It does not predict Short setups.

---

## 🛠️ Tech Stack

- **Core Logic:** Python 3.9+
- **Machine Learning:** XGBoost
- **Data Feed:** `yfinance` (Live Futures Data)
- **Frontend:** Streamlit
- **Visualization:** Plotly (Interactive Charts)
- **Serialization:** Joblib

---

## 🚀 Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/aberesama/gold-price-predictor
    cd gold-price-predictor
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**

    ```bash
    streamlit run app.py
    ```

---

## ⚠️ Disclaimer

This software is an experimental prototype. Algorithmic trading involves significant risk. The "60% Precision" metric is based on historical backtesting and does not guarantee future results.
**Do not risk real capital based solely on this tool.**
