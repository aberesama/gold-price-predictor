import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Gold AI Swing Trader", layout="wide")

# Constants
MODEL_FILE = 'gold_price_model1.joblib'
DECISION_THRESHOLD = 0.42
LOOKBACK_DAYS = 365 

# --- 2. HELPER FUNCTIONS ---

def get_parkinson_volatility(high, low, window=14):
    """Calculates volatility using High/Low range."""
    const = 1.0 / (4.0 * np.log(2.0))
    rs = (np.log(high / low)) ** 2.0
    return np.sqrt(const * rs.rolling(window=window).mean())

@st.cache_data(ttl=3600)
def fetch_live_data():
    """Fetches live market data. Returns cleaned DataFrame."""
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    tickers = "GC=F DX-Y.NYB ^TNX"
    
    raw_data = yf.download(tickers, start=start_date, progress=False)
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = [f"{col[0]}_{col[1]}" for col in raw_data.columns]
    
    df = pd.DataFrame()
    # Robust column fetching
    df['close'] = raw_data.get('Close_GC=F', raw_data.get('Close_GC=F'))
    df['open']  = raw_data.get('Open_GC=F', raw_data.get('Open_GC=F'))
    df['high']  = raw_data.get('High_GC=F', raw_data.get('High_GC=F'))
    df['low']   = raw_data.get('Low_GC=F', raw_data.get('Low_GC=F'))
    df['dxy_close'] = raw_data.get('Close_DX-Y.NYB', raw_data.get('Close_DX-Y.NYB'))
    df['us10y'] = raw_data.get('Close_^TNX', raw_data.get('Close_^TNX'))
    
    df.sort_index(ascending=True, inplace=True)
    return df

def engineer_features(df):
    """Applies exact training logic."""
    data = df.copy()
    
    # 1. Yield Change
    data['Yield_Change_5d'] = data['us10y'].diff(5)
    
    # 2. Position in Range
    rolling_max = data['high'].rolling(20).max()
    rolling_min = data['low'].rolling(20).min()
    data['Position_in_Range'] = (data['close'] - rolling_min) / (rolling_max - rolling_min)
    
    # 3. Volatility (Trigger) - Named 'Volatility' to match model
    vol = get_parkinson_volatility(data['high'], data['low'])
    data['Volatility'] = vol / vol.rolling(50).mean()
    
    # 4. DXY Correlation (Filter) - Named 'DXY_corr' to match model
    data['DXY_corr'] = data['close'].rolling(20).corr(data['dxy_close'])
    
    return data.dropna()

def interpret_factors(row):
    """Generates plain English explanations for the drivers."""
    explanations = {}
    
    # Yield Interpretation
    if row['Yield_Change_5d'] > 0.05:
        explanations['yield'] = "🔴 Rising Yields are putting pressure on Gold."
    elif row['Yield_Change_5d'] < -0.05:
        explanations['yield'] = "🟢 Falling Yields are acting as rocket fuel."
    else:
        explanations['yield'] = "⚪ Yields are flat (Neutral)."

    # Volatility Interpretation
    if row['Volatility'] < 0.8:
        explanations['vol'] = "⚠️ Volatility is compressed. Expect a violent move soon."
    else:
        explanations['vol'] = "⚪ Normal Volatility regime."

    # Correlation Interpretation
    if row['DXY_corr'] > 0.5:
        explanations['dxy'] = "⚠️ Gold is moving WITH the Dollar (Unusual/Safe Haven)."
    elif row['DXY_corr'] < -0.5:
        explanations['dxy'] = "🟢 Gold is trading normally (inverse to Dollar)."
    else:
        explanations['dxy'] = "⚪ No strong correlation with Dollar right now."
        
    return explanations

# --- 3. APPLICATION LOGIC ---

st.title("🏆 Institutional Gold Swing Model")
st.markdown(f"**Live Forecast** | Threshold: `{DECISION_THRESHOLD*100:.0f}%` Confidence")

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"🚨 Critical Error: '{MODEL_FILE}' not found.")
    st.stop()

with st.spinner("Analyzing Market Physics..."):
    try:
        raw_df = fetch_live_data()
        processed_df = engineer_features(raw_df)
        
        # Latest State
        current_state = processed_df.iloc[-1]
        
        # Prepare Input
        feature_names = ['Yield_Change_5d', 'Position_in_Range', 'Volatility', 'DXY_corr']
        input_data = pd.DataFrame([current_state[feature_names]])
        
        # Predict
        prob_buy = model.predict_proba(input_data)[0][1]
        is_buy_signal = prob_buy >= DECISION_THRESHOLD
        
        # Get Explanations
        insights = interpret_factors(current_state)

    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()

# --- 4. DASHBOARD UI ---

# METRICS ROW
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Gold Price", f"${current_state['close']:.2f}")
with col2:
    if is_buy_signal:
        st.success(f"**BUY SIGNAL** (Conf: {prob_buy:.1%})")
    else:
        st.warning(f"**WAIT / NEUTRAL** (Conf: {prob_buy:.1%})")
with col3:
    st.metric("US 10Y Yield", f"{current_state['us10y']:.2f}%", f"{current_state['Yield_Change_5d']:.3f} (5d)")

st.divider()

# --- ACTION PLAN SECTION (New) ---
st.subheader("📋 The Action Plan")

if is_buy_signal:
    st.success("""
    **Bullish Setup Detected.**
    * **Action:** Enter Long (Buy).
    * **Stop Loss:** Place below the recent swing low (approx. 20-day Low).
    * **Reasoning:** The AI sees a statistical edge for higher prices over the next 5 days.
    """)
else:
    st.warning("""
    **No Edge Detected.**
    * **Action:** Stay in Cash / Do Not Buy.
    * **Reasoning:** The probability of a rally is too low to risk capital. 
    * **Wait:** Let the volatility compress or yields drop before entering.
    """)

st.divider()

# --- DRIVER EXPLANATION SECTION (New) ---
st.subheader("🔍 What is Driving Price?")
c1, c2, c3 = st.columns(3)

with c1:
    st.info(f"**Real Yields**\n\n{insights['yield']}")
with c2:
    st.info(f"**Volatility**\n\n{insights['vol']}")
with c3:
    st.info(f"**Dollar Impact**\n\n{insights['dxy']}")

# --- CHART SECTION ---
st.subheader("Last 20 Days Price Action")

chart_data = processed_df.tail(20)

fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(
    x=chart_data.index,
    open=chart_data['open'],
    high=chart_data['high'],
    low=chart_data['low'],
    close=chart_data['close'],
    name="Gold Price"
))

# Yields Overlay
fig.add_trace(go.Scatter(
    x=chart_data.index, 
    y=chart_data['us10y'], 
    name="US 10Y Yield", 
    line=dict(color='blue', width=1, dash='dot'),
    yaxis='y2'
))

fig.update_layout(
    template="plotly_dark",
    title="Gold vs Yields (Micro Structure)",
    yaxis=dict(title="Gold Price"),
    yaxis2=dict(
        title="Yields %", 
        overlaying='y', 
        side='right', 
        showgrid=False
    ),
    xaxis_rangeslider_visible=False,
    height=500,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)