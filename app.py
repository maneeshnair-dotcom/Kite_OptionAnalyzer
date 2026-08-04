# -*- coding: utf-8 -*-
"""
Multi-Underlying Options ATM Dashboard (Streamlit)
----------------------------------------------------
Lets you pick an underlying (index / equity / currency pair / commodity),
an expiry, and an interval, then shows the LATEST candle per ATM+/-N
contract with Volume / Gann / Peak-Trough columns. Data is only pulled
when you click "Refresh Data" (or on an auto-refresh tick) — changing
filters alone does not trigger a fetch.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    """Naive IST datetime. Kite's API always interprets datetimes as IST,
    but hosted servers often run in UTC, so every 'now' in this app goes
    through this helper instead of datetime.now()."""
    return datetime.now(IST).replace(tzinfo=None)


# ── Config ─────────────────────────────────────────────────────────────
DEFAULT_API_KEY = "k5d3p1syii84wrz9"
DEFAULT_API_SECRET = "your_api_secret_here"  # from "Show API secret" on developers.kite.trade
TOKEN_FILE = "kite_token.txt"

# Segments to search for a given underlying's option chain.
SEGMENTS = ["NFO", "BFO", "CDS", "MCX"]

INTERVAL_MAP = {"15 min": "15minute", "30 min": "30minute", "1 hr": "60minute"}
LOOKBACK_DAYS = {"15 min": 3, "30 min": 5, "1 hr": 10}
WMA_WINDOW = 5
LSMA_WINDOW = 7
VOL_SMA_WINDOW = 10

UNDERLYINGS = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50",
    "BANKEX", "FOCIT", "SENSEX", "SENSEX50",
    "633GS2035", "636GS2031", "648GS2035", "668GS2040", "679GS2034",
    "694GS2036", "723GS2039",
    "EURINR", "GBPINR", "JPYINR", "USDINR",
    "COPPER", "CRUDEOIL", "CRUDEOILM", "GOLD", "GOLDM", "MCXBULLDEX",
    "NATGASMINI", "NATURALGAS", "SILVER", "SILVERM", "ZINC",
    "360ONE", "ABB", "ABCAPITAL", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
    "ADANIPORTS", "ADANIPOWER", "ALKEM", "AMBER", "AMBUJACEM", "ANGELONE",
    "APLAPOLLO", "APOLLOHOSP", "ASHOKLEY", "ASIANPAINT", "ASTRAL", "AUBANK",
    "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG",
    "BAJFINANCE", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BDL", "BEL",
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BLUESTARCO", "BOSCHLTD",
    "BPCL", "BRITANNIA", "BSE", "CAMS", "CANBK", "CDSL", "CGPOWER",
    "CHOLAFIN", "CIPLA", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL",
    "CONCOR", "CROMPTON", "CUMMINSIND", "DABUR", "DALBHARAT", "DELHIVERY",
    "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY", "EICHERMOT", "ETERNAL",
    "FEDERALBNK", "FORCEMOT", "FORTIS", "GAIL", "GLENMARK", "GMRAIRPORT",
    "GODFRYPHLP", "GODREJCP", "GODREJPROP", "GRASIM", "GVT&D", "HAL",
    "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HYUNDAI",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX",
    "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "INOXWIND", "IOC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN",
    "JSWENERGY", "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KAYNES", "KEI",
    "KFINTECH", "KOTAKBANK", "KPITTECH", "LAURUSLABS", "LICHSGFIN", "LICI",
    "LODHA", "LT", "LTF", "LTM", "LUPIN", "M&M", "MANAPPURAM", "MANKIND",
    "MARICO", "MARUTI", "MAXHEALTH", "MAZDOCK", "MCX", "MFSL", "MOTHERSON",
    "MOTILALOFS", "MPHASIS", "MUTHOOTFIN", "NAM-INDIA", "NATIONALUM",
    "NAUKRI", "NBCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NYKAA",
    "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI", "PAYTM",
    "PERSISTENT", "PETRONET", "PFC", "PGEL", "PHOENIXLTD", "PIDILITIND",
    "PIIND", "PNB", "PNBHOUSING", "POLICYBZR", "POLYCAB", "POWERGRID",
    "POWERINDIA", "PREMIERENE", "PRESTIGE", "RADICO", "RBLBANK", "RECLTD",
    "RELIANCE", "RVNL", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SRF", "SUNPHARMA",
    "SUPREMEIND", "SUZLON", "SWIGGY", "TATACONSUM", "TATAELXSI",
    "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TIINDIA", "TITAN", "TMPV",
    "TORNTPHARM", "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNIONBANK",
    "UNITDSPR", "UNOMINDA", "UPL", "VBL", "VEDL", "VMM", "VOLTAS",
    "WAAREEENER", "WIPRO", "YESBANK", "ZYDUSLIFE",
]

st.set_page_config(page_title="Options ATM Dashboard", layout="wide")
st.markdown("""
<style>
[data-testid="stTable"] table { font-size: 12px; }
[data-testid="stTable"] th, [data-testid="stTable"] td { padding: 2px 8px !important; }
div.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Credentials ────────────────────────────────────────────────────────
def get_credentials():
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
    if not os.path.exists(TOKEN_FILE):
        return "missing"
    mtime = datetime.fromtimestamp(os.path.getmtime(TOKEN_FILE), tz=IST).replace(tzinfo=None)
    return "fresh" if mtime.date() == now_ist().date() else "stale"


def save_token(access_token: str):
    with open(TOKEN_FILE, "w") as f:
        f.write(access_token)


@st.cache_resource(show_spinner=False)
def get_kite_client(api_key: str, access_token: str) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def handle_kite_login_callback(api_key: str, api_secret: str):
    request_token = st.query_params.get("request_token")
    if not request_token:
        return
    try:
        temp_kite = KiteConnect(api_key=api_key)
        data = temp_kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        save_token(access_token)
        st.session_state["access_token"] = access_token
        get_kite_client.clear()
        st.query_params.clear()
        st.success("Kite token refreshed for today.")
        st.rerun()
    except Exception as e:
        st.query_params.clear()
        st.error(f"Token exchange failed: {e}")


# ── Instrument universe (all segments, cached for an hour) ──────────────
@st.cache_data(ttl=3600, show_spinner="Loading instrument list...")
def get_raw_instruments(_kite) -> pd.DataFrame:
    frames = []
    for seg in SEGMENTS:
        try:
            df = pd.DataFrame(_kite.instruments(seg))
            if not df.empty:
                frames.append(df)
        except Exception as e:
            st.sidebar.warning(f"Could not fetch {seg} instruments: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["expiry"] = pd.to_datetime(combined["expiry"], errors="coerce")
    return combined


def get_underlying_options(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    if raw.empty:
        return raw
    return raw[(raw["name"] == name) & (raw["instrument_type"].isin(["CE", "PE"]))].copy()


def get_underlying_futures(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    if raw.empty:
        return raw
    return raw[(raw["name"] == name) & (raw["instrument_type"] == "FUT")].copy()


def get_available_expiries(opts_for_name: pd.DataFrame) -> list:
    if opts_for_name.empty:
        return []
    today = pd.Timestamp(now_ist().date())
    return sorted(opts_for_name.loc[opts_for_name["expiry"] >= today, "expiry"].dt.date.unique())


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ltp(_kite, exchange: str, tradingsymbol: str):
    key = f"{exchange}:{tradingsymbol}"
    try:
        return _kite.ltp([key])[key]["last_price"]
    except Exception:
        return None


def get_reference_price(_kite, futs_for_name: pd.DataFrame, opts_for_name: pd.DataFrame, expiry_date):
    """Front-month futures LTP as a spot proxy (works uniformly across
    equities/indices/currencies/commodities). Falls back to the median
    strike of the selected expiry if no futures contract is listed."""
    if not futs_for_name.empty:
        today = pd.Timestamp(now_ist().date())
        fut = futs_for_name[futs_for_name["expiry"] >= today].sort_values("expiry")
        if not fut.empty:
            row = fut.iloc[0]
            price = fetch_ltp(_kite, row["exchange"], row["tradingsymbol"])
            if price:
                return price, "future"
    if not opts_for_name.empty and expiry_date is not None:
        strikes = opts_for_name.loc[opts_for_name["expiry"].dt.date == expiry_date, "strike"]
        if not strikes.empty:
            return float(strikes.median()), "strike-median (approx, no futures found)"
    return None, None


def compute_strike_step(strikes: pd.Series) -> float:
    uniq = sorted(strikes.unique())
    diffs = [round(b - a, 4) for a, b in zip(uniq[:-1], uniq[1:]) if b > a]
    if not diffs:
        return 1.0
    return Counter(diffs).most_common(1)[0][0]


def get_atm_chain(opts_for_name: pd.DataFrame, spot: float, atm_range: int, expiry_date) -> pd.DataFrame:
    chain = opts_for_name[opts_for_name["expiry"].dt.date == expiry_date].copy()
    if chain.empty or spot is None:
        return pd.DataFrame()
    step = compute_strike_step(chain["strike"])
    atm_strike = round(spot / step) * step
    lo, hi = atm_strike - atm_range * step, atm_strike + atm_range * step
    chain = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)]
    chain = chain.sort_values(["strike", "instrument_type"]).reset_index(drop=True)
    chain.attrs["atm_strike"] = atm_strike
    chain.attrs["step"] = step
    return chain[["instrument_token", "tradingsymbol", "strike", "instrument_type", "expiry"]]


# ── Indicator functions ──────────────────────────────────────────────
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


def highlight_row(row):
    if row.get("Gann_Reversal_Zone") == "Resistance":
        return ["background-color: #4B1217"] * len(row)
    if row.get("Gann_Reversal_Zone") == "Support":
        return ["background-color: #1A4731"] * len(row)
    return [""] * len(row)


# ── Historical fetch (only called when the user actually triggers it) ───
@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_bucket(_kite, cache_key: str, chain: pd.DataFrame, interval_label: str) -> pd.DataFrame:
    interval = INTERVAL_MAP[interval_label]
    days = LOOKBACK_DAYS[interval_label]
    to_date = now_ist()
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
        "Strike", "Type", "Diff_Peak", "Diff_Trough", "Volume_Signal", "Volume_Trend",
        "Tradingsymbol", "Expiry", "Date",
        "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
        "WMA", "LSMA", "LSMA-WMA",
        "Gann_Resistance", "Gann_Support", "Gann_Reversal_Zone",
        "Volume_SMA", "Volume_Ratio", "OBV",
    ]
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    whole_cols = [c for c in numeric_cols if c != "Volume_Ratio"]
    result[whole_cols] = result[whole_cols].round(0).astype("Int64")
    if "Volume_Ratio" in numeric_cols:
        result["Volume_Ratio"] = result["Volume_Ratio"].round(2)
    return result[cols].reset_index(drop=True)


# ── UI ────────────────────────────────────────────────────────────────
api_key, api_secret, access_token = get_credentials()
handle_kite_login_callback(api_key, api_secret)
if "access_token" in st.session_state:
    access_token = st.session_state["access_token"]

with st.sidebar:
    st.header("Settings")
    underlying = st.selectbox("Underlying", UNDERLYINGS, index=UNDERLYINGS.index("NIFTY"))
    interval_label = st.selectbox("Interval", list(INTERVAL_MAP.keys()), index=0)
    atm_range = st.number_input("ATM ± strikes", min_value=1, max_value=10, value=5)

    st.divider()
    refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True, type="primary")
    auto_run = st.checkbox("Auto refresh", value=False)
    refresh_secs = st.slider("Refresh every (sec)", 15, 300, 60, step=15, disabled=not auto_run)
    if auto_run:
        if AUTOREFRESH_OK:
            st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh")
        else:
            st.warning("Install `streamlit-autorefresh` (see requirements.txt) for auto refresh to work.")
    st.caption("Data is pulled only when you click Refresh, or on each auto-refresh tick — changing filters alone will not fetch.")

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
    st.caption("Opens Zerodha's login in a new tab and captures the token automatically on redirect.")
    if not access_token:
        access_token = st.text_input("Or paste an access token manually", type="password")

# Build the client only now, so a token typed above this run is already reflected.
kite = get_kite_client(api_key, access_token) if access_token else None

if not kite:
    st.title("Options ATM Dashboard")
    st.info("Log in via the sidebar button (or paste a token) to load the dashboard.")
    st.stop()

raw = get_raw_instruments(kite)
opts_for_name = get_underlying_options(raw, underlying)
futs_for_name = get_underlying_futures(raw, underlying)
expiries = get_available_expiries(opts_for_name)

# ── Top row: title left, expiry selector top-right ──────────────────
col_title, col_expiry = st.columns([4, 1])
with col_title:
    st.title(f"{underlying} — ATM ±{atm_range} Dashboard")
with col_expiry:
    expiry_date = st.selectbox(
        "Expiry", expiries, index=0, disabled=not expiries,
        format_func=lambda d: d.strftime("%d-%b-%Y (%a)"),
        label_visibility="visible",
    ) if expiries else None

if not expiries:
    st.warning(f"No option contracts found for '{underlying}' across {', '.join(SEGMENTS)}.")
    st.stop()

spot, spot_source = get_reference_price(kite, futs_for_name, opts_for_name, expiry_date)
chain = get_atm_chain(opts_for_name, spot, atm_range, expiry_date)
atm_strike = chain.attrs.get("atm_strike")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reference Price", f"{spot:,.2f}" if spot else "—",
          help="Front-month futures LTP used as an ATM anchor (not the literal index/spot price).")
c2.metric("ATM Strike", atm_strike if atm_strike else "—")
c3.metric("Expiry", str(expiry_date))
c4.metric("Interval", interval_label)
if spot_source and spot_source != "future":
    st.caption(f"⚠ Reference price source: {spot_source}")

# ── Gate the expensive per-contract fetch behind the button/auto-refresh ─
if "latest_data" not in st.session_state:
    st.session_state.latest_data = pd.DataFrame()
    st.session_state.latest_meta = {}

do_fetch = refresh_clicked or auto_run
if do_fetch and not chain.empty:
    with st.spinner("Fetching latest data..."):
        cache_key = f"{underlying}-{expiry_date}-{atm_range}-{interval_label}"
        st.session_state.latest_data = fetch_latest_bucket(kite, cache_key, chain, interval_label)
        st.session_state.latest_meta = {
            "underlying": underlying, "expiry": expiry_date,
            "interval": interval_label, "time": now_ist(),
        }

data = st.session_state.latest_data
meta = st.session_state.latest_meta

if data.empty:
    st.info("Click '🔄 Refresh Data' in the sidebar to load the option chain.")
else:
    st.caption(
        f"Showing {meta.get('underlying')} · {meta.get('expiry')} · {meta.get('interval')} "
        f"· last refreshed {meta.get('time').strftime('%H:%M:%S')} IST"
    )
    st.table(data.style.apply(highlight_row, axis=1))

    signals = data[(data["Volume_Signal"] != "") | (data["Diff_Peak"] != "") | (data["Diff_Trough"] != "")]
    if not signals.empty:
        st.subheader("Active signals this bucket")
        st.table(signals.style.apply(highlight_row, axis=1))
