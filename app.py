# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Optional libs
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Multi-Coin Crypto Intelligence")
st.title("🚀 Multi-Coin Crypto Intelligence — Dynamic Dashboard")

# ---------------- Helpers ----------------
@st.cache_data
def load_df(path):
    import csv
    import os
    import pandas as pd
    import numpy as np

    if not os.path.exists(path):
        return pd.DataFrame()

    # -------- Detect delimiter --------
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
            delimiter = csv.Sniffer().sniff(sample).delimiter
    except Exception:
        delimiter = ","

    df = pd.read_csv(path, sep=delimiter, low_memory=False)

    # -------- Detect timestamp column --------
    possible_time_cols = [
        "open_time", "Open Time", "timestamp",
        "time", "date", "datetime", "Date"
    ]

    time_col = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    # If not found by name, auto-detect by pattern
    if time_col is None:
        for col in df.columns:
            sample_vals = df[col].astype(str).head(20)
            if sample_vals.str.contains(r"\d{4}-\d{2}-\d{2}").any() or \
               sample_vals.str.contains(r"\d{2}-\d{2}-\d{4}").any():
                time_col = col
                break

    if time_col is None:
        return pd.DataFrame()

    df = df.rename(columns={time_col: "open_time"})
    df["open_time"] = df["open_time"].astype(str).str.strip()

    # -------- Try multiple datetime parsing strategies --------
    parse_attempts = [
        lambda x: pd.to_datetime(x, format="%d-%m-%Y %H.%M", errors="coerce"),
        lambda x: pd.to_datetime(x, format="%d-%m-%Y %H:%M:%S", errors="coerce"),
        lambda x: pd.to_datetime(x, errors="coerce"),
        lambda x: pd.to_datetime(x, dayfirst=True, errors="coerce"),
    ]

    best_parsed = None
    best_valid = 0

    for attempt in parse_attempts:
        parsed = attempt(df["open_time"])
        valid_count = parsed.notna().sum()
        if valid_count > best_valid:
            best_valid = valid_count
            best_parsed = parsed
        if valid_count >= len(df) * 0.9:
            break

    # Try UNIX timestamps if still failing
    if best_valid == 0:
        numeric = pd.to_numeric(df["open_time"], errors="coerce")
        if numeric.notna().sum() > 0:
            if numeric.mean() > 1e12:
                best_parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
            else:
                best_parsed = pd.to_datetime(numeric, unit="s", errors="coerce")

    if best_parsed is None:
        return pd.DataFrame()

    df["open_time"] = best_parsed
    df = df.dropna(subset=["open_time"])

    if df.empty:
        return pd.DataFrame()

    # Remove timezone if present
    try:
        df["open_time"] = df["open_time"].dt.tz_localize(None)
    except Exception:
        pass

    df = df.set_index("open_time").sort_index()

    # Clean numeric issues
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df
def try_resample(df_in, rule):
    """Resample df_in using sensible aggregations. Return resampled or original on failure."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "sentiment_lag1": "mean",
        "Volatility": "mean",
        "Buy Ratio": "mean",
        "Return %": "mean",
        "MA20": "mean"
    }
    agg = {k: v for k, v in agg.items() if k in df_in.columns}
    try:
        out = df_in.resample(rule).agg(agg).dropna()
        return out
    except Exception:
        return df_in

def build_future_index(last_ts, steps, freq):
    """Create a future DatetimeIndex; fallback to daily if freq unsupported."""
    try:
        return pd.date_range(start=last_ts, periods=steps + 1, freq=freq)[1:]
    except Exception:
        return pd.date_range(start=last_ts, periods=steps + 1, freq="D")[1:]

def safe_last(series):
    try:
        return series.iloc[-1]
    except Exception:
        return None

# ---------------- Load CSVs ----------------
# Update these paths to where your CSVs actually are.
paths = {
    "BTC": "data/df1.csv",
    "ETH": "data/df2.csv",
    "USDT": "data/df3.csv",
    "SOL": "data/df4.csv",
    "BNB": "data/df5.csv"
}

datasets = {}
for key, p in paths.items():
    try:
        datasets[key] = load_df(p)
    except Exception:
        datasets[key] = pd.DataFrame()

# ---------------- Sidebar controls ----------------
coin = st.sidebar.selectbox("Select Coin", list(datasets.keys()))
df_full = datasets[coin]

if df_full.empty:
    st.error(f"{coin} data missing or empty. Check '{paths[coin]}'")
    st.stop()

# Build month list bounded by available data (up to last 24 months)
end_month = df_full.index.max().to_period("M")
start_candidate = end_month - 23
available_start = df_full.index.min().to_period("M")
start_month = max(start_candidate, available_start)
all_months = pd.period_range(start=start_month, end=end_month, freq="M")
month_strs = [str(m) for m in all_months]

# session state keys (keeps per-session behavior simple and not sticky across coins)
if "selected_coin" not in st.session_state:
    st.session_state.selected_coin = coin
if "selected_month" not in st.session_state or st.session_state.selected_coin != coin:
    # default to most recent available month for this coin
    st.session_state.selected_coin = coin
    st.session_state.selected_month = month_strs[-1]
if "resample_rule" not in st.session_state:
    st.session_state.resample_rule = "1D"

# Sidebar inputs (user can change these)
selected_month_input = st.sidebar.selectbox("Select Month (last 24 months)", month_strs, index=len(month_strs) - 1)
resample_input = st.sidebar.selectbox("Resample Timeframe", ["15T", "1H", "1D"], index=2)
forecast_horizon = st.sidebar.slider("Forecast Horizon (steps)", min_value=5, max_value=60, value=14)

# Explicit apply/reset controls
apply_filter = st.sidebar.button("Apply Filter")
reset_to_recent = st.sidebar.button("Reset to Most Recent")

# Apply / reset logic
if reset_to_recent:
    st.session_state.selected_month = month_strs[-1]
    st.session_state.resample_rule = resample_input

if apply_filter:
    st.session_state.selected_month = selected_month_input
    st.session_state.resample_rule = resample_input

# When user switches coin in the sidebar, update session defaults immediately
if st.session_state.selected_coin != coin:
    st.session_state.selected_coin = coin
    st.session_state.selected_month = month_strs[-1]
    st.session_state.resample_rule = resample_input

# ---------------- Filter data for selected month & resample ----------------
selected_month = pd.Period(st.session_state.selected_month, freq="M")
df = df_full[df_full.index.to_period("M") == selected_month].copy()

if df.empty:
    st.warning(f"No data for {st.session_state.selected_month} for {coin}. Try another month or coin.")
    st.stop()

# Resample using selected rule
df = try_resample(df, st.session_state.resample_rule)
if df is None or df.empty:
    st.warning("No data after resampling. Try a different resample timeframe or month.")
    st.stop()

# ---------------- Page selector ----------------
page = st.sidebar.selectbox("Select Page", [
    "Overview",
    "Price Explorer",
    "Forecast",
    "Sentiment Impact",
    "Volatility & Risk",
    "Indicators",
    "Correlations",
    "Strategy Backtest",
    "Interactive Explorer"
])

# ---------------- Pages ----------------

# Overview
if page == "Overview":
    st.header(f"{coin} — Overview ({st.session_state.selected_month})")
    col1, col2, col3 = st.columns(3)
    last_close = safe_last(df["close"]) if "close" in df.columns else None
    col1.metric("Latest Price", f"${last_close:,.2f}" if last_close is not None else "n/a")
    if "Return %" in df.columns:
        col2.metric("Return %", f"{safe_last(df['Return %']):.2f}%")
    else:
        col2.write("Return %: n/a")
    if "sentiment_lag1" in df.columns:
        col3.metric("Sentiment", f"{safe_last(df['sentiment_lag1']):.2f}")
    else:
        col3.write("Sentiment: n/a")
    if "close" in df.columns:
        st.plotly_chart(px.line(df, y="close", title=f"{coin} Close Price — {st.session_state.selected_month}"), use_container_width=True)
    else:
        st.info("No 'close' column to display price.")

# Price Explorer
elif page == "Price Explorer":
    st.header(f"{coin} — Price Explorer ({st.session_state.selected_month})")
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"])])
        fig.update_layout(title=f"{coin} Candlestick — {st.session_state.selected_month}", height=650)
        st.plotly_chart(fig, use_container_width=True)
    elif "close" in df.columns:
        st.plotly_chart(px.line(df, y="close", title=f"{coin} Close Price — {st.session_state.selected_month}"), use_container_width=True)
    else:
        st.info("No OHLC or close data available to plot.")

# Forecast
elif page == "Forecast":
    st.header(f"{coin} — Multi-Model Forecast ({st.session_state.selected_month})")

    if "close" not in df.columns:
        st.error("No 'close' column available to forecast.")
    elif len(df) < 10:
        st.warning("Not enough data points to forecast reliably. Try a longer period/resample.")
    else:
        model_choice = st.multiselect("Select Models", ["ARIMA", "SARIMA", "Prophet", "LSTM"], default=["ARIMA"])

        # Advanced options
        with st.expander("Advanced model settings"):
            arima_order_input = st.text_input("ARIMA order p,d,q", value="5,1,0")
            sarima_order_input = st.text_input("SARIMA p,d,q,P,D,Q,s", value="1,1,1,1,1,1,12")

        # parse orders with safe fallbacks
        try:
            p, d, q = [int(x.strip()) for x in arima_order_input.split(",")]
        except Exception:
            p, d, q = 5, 1, 0
        try:
            parts = [int(x.strip()) for x in sarima_order_input.split(",")]
            sar_order = tuple(parts[:3])
            sar_seasonal = tuple(parts[3:7])
        except Exception:
            sar_order = (1, 1, 1)
            sar_seasonal = (1, 1, 1, 12)

        future_index = build_future_index(df.index[-1], forecast_horizon, st.session_state.resample_rule)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["close"],
            name="Historical",
            line=dict(color="#A0C4FF", width=2)
        ))

        # ARIMA
        if "ARIMA" in model_choice:
            try:
                with st.spinner("Fitting ARIMA..."):
                    arima_m = ARIMA(df["close"], order=(p, d, q))
                    arima_f = arima_m.fit()
                    arima_pred = arima_f.forecast(steps=forecast_horizon)
                    fig.add_trace(go.Scatter(x=future_index, y=arima_pred, name=f"ARIMA({p},{d},{q})"))
            except Exception as e:
                st.error("ARIMA failed: " + str(e))

        # SARIMA
        if "SARIMA" in model_choice:
            try:
                with st.spinner("Fitting SARIMA..."):
                    sar_m = SARIMAX(df["close"], order=sar_order, seasonal_order=sar_seasonal)
                    sar_f = sar_m.fit(disp=False)
                    sar_pred = sar_f.forecast(steps=forecast_horizon)
                    fig.add_trace(go.Scatter(x=future_index, y=sar_pred, name=f"SARIMA{sar_order}x{sar_seasonal}"))
            except Exception as e:
                st.error("SARIMA failed: " + str(e))

        # Prophet
        if "Prophet" in model_choice:
            if not PROPHET_AVAILABLE:
                st.warning("Prophet not installed in this environment; skipping Prophet.")
            else:
                try:
                    with st.spinner("Fitting Prophet..."):
                        # prepare prophet df: reset_index ensures we have a datetime column
                        pdf = df.reset_index().iloc[:, :2].copy()
                        pdf.columns = ["ds", "y"]
                        # make tz-naive
                        pdf["ds"] = pd.to_datetime(pdf["ds"]).dt.tz_localize(None)
                        model_p = Prophet()
                        model_p.fit(pdf)
                        # Prophet sometimes fails on sub-daily freq in some setups; try requested freq and fallback to daily
                        try:
                            future = model_p.make_future_dataframe(periods=forecast_horizon, freq=st.session_state.resample_rule)
                            forecast = model_p.predict(future)
                            prophet_pred = forecast["yhat"].iloc[-forecast_horizon:].values
                            # trim/pad to match future_index length if necessary
                            prophet_pred = prophet_pred[-len(future_index):]
                            fig.add_trace(go.Scatter(x=future_index, y=prophet_pred, name="Prophet"))
                        except Exception:
                            future = model_p.make_future_dataframe(periods=forecast_horizon, freq="D")
                            forecast = model_p.predict(future)
                            prophet_pred = forecast["yhat"].iloc[-forecast_horizon:].values
                            fallback_index = build_future_index(df.index[-1], forecast_horizon, "D")
                            fig.add_trace(go.Scatter(x=fallback_index, y=prophet_pred, name="Prophet (daily fallback)"))
                except Exception as e:
                    st.error("Prophet failed: " + str(e))

        # LSTM
        if "LSTM" in model_choice:
            if not TENSORFLOW_AVAILABLE:
                st.warning("TensorFlow / sklearn not installed; skipping LSTM.")
            else:
                try:
                    with st.spinner("Training LSTM..."):
                        arr = df[["close"]].values.astype("float32")
                        scaler = MinMaxScaler()
                        scaled = scaler.fit_transform(arr)
                        window = min(30, len(scaled) - 1)
                        X, y = [], []
                        for i in range(window, len(scaled)):
                            X.append(scaled[i - window:i, 0])
                            y.append(scaled[i, 0])
                        X = np.array(X)
                        y = np.array(y)
                        if X.size == 0:
                            raise ValueError("Not enough rows for LSTM windows.")
                        X = X.reshape((X.shape[0], X.shape[1], 1))
                        model_l = Sequential()
                        model_l.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
                        model_l.add(Dense(1))
                        model_l.compile(optimizer="adam", loss="mse")
                        model_l.fit(X, y, epochs=10, batch_size=16, verbose=0)
                        last_window = scaled[-window:, 0]
                        preds_scaled = []
                        for _ in range(forecast_horizon):
                            p = model_l.predict(last_window.reshape(1, window, 1), verbose=0)[0][0]
                            preds_scaled.append([p])
                            last_window = np.append(last_window[1:], p)
                        preds = scaler.inverse_transform(np.array(preds_scaled)).flatten()
                        fig.add_trace(go.Scatter(x=future_index, y=preds, name="LSTM"))
                except Exception as e:
                    st.error("LSTM failed: " + str(e))

        fig.update_layout(title=f"{coin} Forecast Comparison — {st.session_state.selected_month}", height=650)
        st.plotly_chart(fig, use_container_width=True)

# Sentiment Impact
elif page == "Sentiment Impact":
    st.header(f"{coin} — Sentiment Impact ({st.session_state.selected_month})")
    fig = go.Figure()
    if "sentiment_lag1" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sentiment_lag1"], name="Sentiment"))
    if "close" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Price", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)

# Volatility & Risk
elif page == "Volatility & Risk":
    st.header(f"{coin} — Volatility & Risk ({st.session_state.selected_month})")
    if "Volatility" in df.columns:
        st.plotly_chart(px.line(df, y="Volatility", title="Volatility"), use_container_width=True)
    else:
        st.info("No 'Volatility' column.")
    if "Buy Ratio" in df.columns:
        st.plotly_chart(px.line(df, y="Buy Ratio", title="Buy Ratio"), use_container_width=True)

# Indicators
elif page == "Indicators":
    st.header(f"{coin} — Indicators ({st.session_state.selected_month})")
    if "MA20" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
        st.plotly_chart(fig, use_container_width=True)
    elif "close" in df.columns:
        df["MA20_calc"] = df["close"].rolling(window=20, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20_calc"], name="MA20 (calc)"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'close' column to compute indicators.")

# Correlations
elif page == "Correlations":
    st.header(f"{coin} — Correlations ({st.session_state.selected_month})")
    candidate = ["close", "volume", "sentiment_lag1", "Buy Ratio", "Volatility", "Return %"]
    cols = [c for c in candidate if c in df.columns]
    if len(cols) >= 2:
        corr = df[cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Matrix"), use_container_width=True)
    else:
        st.info("Not enough columns for correlation. Need two of: " + ", ".join(candidate))

# Strategy Backtest
elif page == "Strategy Backtest":
    st.header(f"{coin} — Strategy Backtest ({st.session_state.selected_month})")
    if "Return %" in df.columns and "sentiment_lag1" in df.columns:
        data = df.copy()
        data["strategy"] = np.where(data["sentiment_lag1"] > data["sentiment_lag1"].median(), data["Return %"]/100, -data["Return %"]/100)
        data["equity"] = (1 + data["strategy"]).cumprod()
        st.plotly_chart(px.line(data, y="equity", title="Strategy Equity"), use_container_width=True)
        st.dataframe(data[["Return %", "sentiment_lag1", "strategy"]].tail(20))
    else:
        st.info("Need 'Return %' and 'sentiment_lag1' for backtest.")

# Interactive Explorer
elif page == "Interactive Explorer":
    st.header(f"{coin} — Interactive Explorer ({st.session_state.selected_month})")
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    date_range = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    if isinstance(date_range, list) and len(date_range) == 2:
        sub = df.loc[str(date_range[0]):str(date_range[1])]
    else:
        sub = df
    st.dataframe(sub.tail(200))
    if "close" in sub.columns:
        st.plotly_chart(px.line(sub, y="close", title="Close Price (selected range)"), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Tip: choose coin → choose month → click Apply Filter if you want to pin selection. Prophet and LSTM are optional (install packages to enable).")