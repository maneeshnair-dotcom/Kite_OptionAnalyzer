# -*- coding: utf-8 -*-
"""
Nifty Options ATM+-5 Dashboard (Streamlit)
-------------------------------------------
Shows only the LATEST candle per contract for the near-expiry NIFTY option
chain (ATM +/- 5 strikes), with Volume / Gann / Peak-Trough columns computed
the same way as nifty_options_atm_gann.py. User can switch interval (15min /
30min / 1hr) and turn on auto-refresh.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push app.py + requirements.txt to a GitHub repo.
    2. On share.streamlit.io, point to the repo and app.py.
    3. Add KITE_API_KEY and KITE_ACCESS_TOKEN under the app's Settings ->
       Secrets (see the sidebar note below — the access token still has to
       be regenerated once per trading day; the cloud app can't run the
       local login flow for you).
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view
from kiteconnect import KiteConnect

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

# ── Config ─────────────────────────────────────────────────────────────
DEFAULT_API_KEY = "k5d3p1syii84wrz9"
DEFAULT_API_SECRET = "your_api_secret_here"  # from "Show API secret" on developers.kite.trade
TOKEN_FILE = "kite_token.txt"
STRIKE_STEP = 50
INTERVAL_MAP = {"15 min": "15minute", "30 min": "30minute", "1 hr": "60minute"}
LOOKBACK_DAYS = {"15 min": 3, "30 min": 5, "1 hr": 10}   # enough bars for WMA/LSMA warmup
WMA_WINDOW = 5
LSMA_WINDOW = 7
VOL_SMA_WINDOW = 10

st.set_page_config(page_title="Nifty Options ATM Dashboard", layout="wide")


# ── Credentials ────────────────────────────────────────────────────────
def get_credentials():
    """Priority: Streamlit secrets -> local kite_token.txt -> manual sidebar input."""
    api_key = st.secrets.get("KITE_API_KEY", DEFAULT_API_KEY) if hasattr(st, "secrets") else DEFAULT_API_KEY
    api_secret = st.secrets.get("KITE_API_SECRET", DEFAULT_API_SECRET) if hasattr(st, "secrets") else DEFAULT_API_SECRET
    access_token = None
    if hasattr(st, "secrets") and "KITE_ACCESS_TOKEN" in st.secrets:
        access_token = st.secrets["KITE_ACCESS_TOKEN"]
    elif os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            access_token = f.read().strip()
    return api_key, api_secret, access_token


def token_status() -> str:
    """Rough freshness check: Kite access tokens invalidate every day (~6 AM),
    so a token file not written today is almost certainly stale."""
    if not os.path.exists(TOKEN_FILE):
        return "missing"
    mtime = datetime.fromtimestamp(os.path.getmtime(TOKEN_FILE))
    return "fresh" if mtime.date() == datetime.now().date() else "stale"


def save_token(access_token: str):
    with open(TOKEN_FILE, "w") as f:
        f.write(access_token)


@st.cache_resource(show_spinner=False)
def get_kite_client(api_key: str, access_token: str) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def handle_kite_login_callback(api_key: str, api_secret: str):
    """If Kite has just redirected back here with a request_token, exchange it
    for an access_token, persist it, and clear the query param."""
    request_token = st.query_params.get("request_token")
    if not request_token:
        return
    try:
        temp_kite = KiteConnect(api_key=api_key)
        data = temp_kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        save_token(access_token)
        st.session_state["access_token"] = access_token
        get_kite_client.clear()  # drop cached client so the new token is picked up
        st.query_params.clear()
        st.success("Kite token refreshed for today.")
        st.rerun()
    except Exception as e:
        st.query_params.clear()
        st.error(f"Token exchange failed: {e}")


# ── Indicator functions (same logic as nifty_options_atm_gann.py) ───────
def calculate_wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    denom = weights.sum()
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) >= window:
        windows = sliding_window_view(arr, window)
        out[window - 1:] = windows @ weights / denom
    return pd.Series(out, index=series.index)


def calculate_lsma(series: pd.Series, window: int) -> pd.Series:
    n = window
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        x = np.arange(n, dtype=float)
        sum_x, sum_x2 = x.sum(), (x ** 2).sum()
        denom = n * sum_x2 - sum_x ** 2
        windows = sliding_window_view(arr, n)
        sum_y = windows.sum(axis=1)
        sum_xy = windows @ x
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        out[n - 1:] = slope * (n - 1) + intercept
    return pd.Series(out, index=series.index)


def calculate_gann_targets(df: pd.DataFrame, zone_tolerance: float = 0.018) -> pd.DataFrame:
    close = df["CLOSE"]
    sqrt_p = np.sqrt(close.clip(lower=0.01))
    upside = np.ceil(sqrt_p) ** 2
    downside = np.floor(sqrt_p) ** 2
    df["Gann_Resistance"] = upside.round(2)
    df["Gann_Support"] = downside.round(2)
    up_ratio = (close - upside).abs() / upside.replace(0, np.nan)
    down_ratio = (close - downside).abs() / downside.replace(0, np.nan)
    df["Gann_Reversal_Zone"] = np.select(
        [up_ratio < zone_tolerance, down_ratio < zone_tolerance],
        ["Resistance", "Support"], default="",
    )
    return df


def volume_analysis(df: pd.DataFrame, window: int = VOL_SMA_WINDOW) -> pd.DataFrame:
    df["Volume_SMA"] = df["VOLUME"].rolling(window).mean()
    df["Volume_Ratio"] = (df["VOLUME"] / df["Volume_SMA"]).round(2)
    df["OBV"] = (np.sign(df["CLOSE"].diff()) * df["VOLUME"]).fillna(0).cumsum()
    df["Volume_Trend"] = np.where(df["VOLUME"] > df["Volume_SMA"], "Rising", "Falling")
    df["Volume_Signal"] = ""
    strong_vol = df["Volume_Ratio"] > 1.8
    df.loc[strong_vol & (df["CLOSE"] > df["OPEN"]), "Volume_Signal"] = "Strong_Buy_Vol"
    df.loc[strong_vol & (df["CLOSE"] < df["OPEN"]), "Volume_Signal"] = "Strong_Sell_Vol"
    return df


def add_peak_trough(df: pd.DataFrame) -> pd.DataFrame:
    diff = df["LSMA"] - df["WMA"]
    df["LSMA-WMA"] = diff.round(2)
    df["Diff_Peak"] = ((diff < diff.shift(1)) & (diff.shift(1) > diff.shift(2))).map({True: "Peak", False: ""})
    df["Diff_Trough"] = ((diff > diff.shift(1)) & (diff.shift(1) < diff.shift(2))).map({True: "Trough", False: ""})
    return df


# ── Chain + data fetch (cached per interval, TTL matches refresh cadence) ─
@st.cache_data(ttl=60, show_spinner=False)
def get_atm_chain(_kite, atm_range: int) -> pd.DataFrame:
    instruments = pd.DataFrame(_kite.instruments("NFO"))
    opts = instruments[(instruments["name"] == "NIFTY") & (instruments["segment"] == "NFO-OPT")].copy()
    opts["expiry"] = pd.to_datetime(opts["expiry"])
    today = pd.Timestamp.now().normalize()
    nearest_expiry = opts.loc[opts["expiry"] >= today, "expiry"].min()
    chain = opts[opts["expiry"] == nearest_expiry].copy()

    spot = _kite.quote(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    atm_strike = round(spot / STRIKE_STEP) * STRIKE_STEP
    lo, hi = atm_strike - atm_range * STRIKE_STEP, atm_strike + atm_range * STRIKE_STEP

    chain = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)]
    chain = chain[chain["instrument_type"].isin(["CE", "PE"])]
    chain = chain.sort_values(["strike", "instrument_type"]).reset_index(drop=True)
    chain.attrs["spot"] = spot
    chain.attrs["atm_strike"] = atm_strike
    return chain[["instrument_token", "tradingsymbol", "strike", "instrument_type", "expiry"]]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_bucket(_kite, chain_key: str, chain: pd.DataFrame, interval_label: str) -> pd.DataFrame:
    interval = INTERVAL_MAP[interval_label]
    days = LOOKBACK_DAYS[interval_label]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    rows = []
    for _, row in chain.iterrows():
        try:
            candles = _kite.historical_data(row["instrument_token"], from_date, to_date, interval)
        except Exception as e:
            st.warning(f"{row['tradingsymbol']}: historical fetch failed ({e})")
            continue
        if not candles:
            continue
        df = pd.DataFrame(candles).rename(columns={
            "date": "Date", "open": "OPEN", "high": "HIGH",
            "low": "LOW", "close": "CLOSE", "volume": "VOLUME",
        })
        df["WMA"] = calculate_wma(df["CLOSE"], WMA_WINDOW)
        df["LSMA"] = calculate_lsma(df["CLOSE"], LSMA_WINDOW)
        df = add_peak_trough(df)
        df = calculate_gann_targets(df)
        df = volume_analysis(df)

        latest = df.iloc[-1].copy()
        latest["Tradingsymbol"] = row["tradingsymbol"]
        latest["Strike"] = row["strike"]
        latest["Type"] = row["instrument_type"]
        latest["Expiry"] = row["expiry"].date()
        rows.append(latest)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    cols = [
        "Tradingsymbol", "Strike", "Type", "Expiry", "Date",
        "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
        "WMA", "LSMA", "LSMA-WMA", "Diff_Peak", "Diff_Trough",
        "Gann_Resistance", "Gann_Support", "Gann_Reversal_Zone",
        "Volume_SMA", "Volume_Ratio", "OBV", "Volume_Trend", "Volume_Signal",
    ]
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(2)
    return result[cols].reset_index(drop=True)


def highlight_row(row):
    if row["Gann_Reversal_Zone"] == "Resistance":
        return ["background-color: #4B1217"] * len(row)
    if row["Gann_Reversal_Zone"] == "Support":
        return ["background-color: #1A4731"] * len(row)
    return [""] * len(row)


# ── UI ────────────────────────────────────────────────────────────────
st.title("NIFTY Options — ATM ±5 Dashboard")

api_key, api_secret, access_token = get_credentials()

# Process a Kite login redirect (request_token in the URL) before anything else.
handle_kite_login_callback(api_key, api_secret)

# A token generated by clicking the button this session takes priority.
if "access_token" in st.session_state:
    access_token = st.session_state["access_token"]

with st.sidebar:
    st.header("Settings")
    interval_label = st.selectbox("Interval", list(INTERVAL_MAP.keys()), index=0)
    atm_range = st.number_input("ATM ± strikes", min_value=1, max_value=10, value=5)
    auto_run = st.checkbox("Auto refresh", value=False)
    refresh_secs = st.slider("Refresh every (sec)", 15, 300, 60, step=15, disabled=not auto_run)

    if auto_run:
        if AUTOREFRESH_OK:
            st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh")
        else:
            st.warning("Install `streamlit-autorefresh` (see requirements.txt) for auto refresh to work.")

    st.divider()
    st.subheader("Kite Login")
    status = token_status()
    if status == "fresh":
        st.success("Token generated today ✅")
    elif status == "stale":
        st.warning("Token is from a previous day — refresh it below.")
    else:
        st.warning("No token found — log in below.")

    temp_kite = KiteConnect(api_key=api_key)
    st.link_button("🔐 Login to Kite / Refresh Token", temp_kite.login_url(), use_container_width=True)
    st.caption(
        "Opens Zerodha's login in a new tab. After you approve, it redirects back "
        "here and the token is captured and saved automatically — no copy/pasting."
    )

    if not access_token:
        access_token = st.text_input("Or paste an access token manually", type="password")

if not access_token:
    st.info("Log in via the sidebar button (or paste a token) to load the dashboard.")
    st.stop()

kite = get_kite_client(api_key, access_token)

try:
    chain = get_atm_chain(kite, atm_range)
except Exception as e:
    st.error(f"Failed to load option chain: {e}")
    st.stop()

spot = chain.attrs.get("spot")
atm_strike = chain.attrs.get("atm_strike")
expiry = chain["expiry"].iloc[0].date() if not chain.empty else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("NIFTY Spot", f"{spot:,.2f}" if spot else "—")
c2.metric("ATM Strike", atm_strike or "—")
c3.metric("Expiry", str(expiry) if expiry else "—")
c4.metric("Interval", interval_label)

data = fetch_latest_bucket(kite, f"{expiry}-{atm_range}", chain, interval_label)

if data.empty:
    st.warning("No data returned — check Historical Data API permission on your Kite Connect app.")
else:
    st.caption(f"Latest bucket per contract · last updated {datetime.now().strftime('%H:%M:%S')}")
    st.dataframe(data.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

    signals = data[(data["Volume_Signal"] != "") | (data["Diff_Peak"] != "") | (data["Diff_Trough"] != "")]
    if not signals.empty:
        st.subheader("Active signals this bucket")
        st.dataframe(signals, use_container_width=True, hide_index=True)
