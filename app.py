import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyotp
from SmartApi.smartConnect import SmartConnect
from datetime import datetime, timedelta

from index_tokens import index_tokens   # ✅ IMPORT FROM FILE

st.set_page_config(page_title="RRG - Angel One Indices", layout="wide")

@st.cache_resource
def angel_login():
    api_key = "EKa93pFu"
    client_id = "R59803990"
    password = "1234"
    totp_secret = "5W4MC6MMLANC3UYOAW2QDUIFEU"
    totp = pyotp.TOTP(totp_secret).now()
    obj = SmartConnect(api_key=api_key)
    obj.generateSession(client_id, password, totp)
    return obj

obj = angel_login()

st.title("📊 Relative Rotation Graph (RRG) — Angel One Index Data")

with st.sidebar:
    st.header("⚙️ Controls")
    days = st.slider("History (days)", 120, 2000, 365, step=30)
    interval = st.selectbox("Interval", ["ONE_DAY", "ONE_HOUR"], index=0)
    window = st.slider("Rolling Window", 5, 30, 14, step=1)
    tail = st.slider("Tail Length", 1, 15, 5, step=1)

    keys = sorted(index_tokens.keys())

    benchmark_key = st.selectbox(
        "Benchmark",
        keys,
        index=keys.index("NIFTY_50") if "NIFTY_50" in keys else 0
    )

    default_sectors = [k for k in ["NIFTY_AUTO","NIFTY_IT","NIFTY_FMCG","NIFTY_METAL","NIFTY_REALTY","NIFTY_BANK"] if k in index_tokens]
    sectors = st.multiselect("Sectors", keys, default=default_sectors)

    if st.button("🧹 Clear Cache / Reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

sectors = [s for s in sectors if s != benchmark_key]
if not sectors:
    st.warning("Select at least 1 sector in sidebar.")
    st.stop()

def fetch_close_series(token: str, days: int, interval: str) -> pd.Series:
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    params = {
        "exchange": "NSE",
        "symboltoken": str(token),
        "interval": interval,
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M"),
    }

    res = obj.getCandleData(params)
    data = res.get("data", None)
    if not data:
        return pd.Series(dtype="float64")

    df = pd.DataFrame(data, columns=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df.set_index("date", inplace=True)
    return df["close"].astype(float)

@st.cache_data(ttl=60*15)
def load_series_map(keys, days, interval):
    out = {}
    for k in keys:
        out[k] = fetch_close_series(index_tokens[k], days, interval)
    return out

with st.spinner("Fetching candle data..."):
    bench_series = fetch_close_series(index_tokens[benchmark_key], days, interval)
    sector_series_map = load_series_map(sectors, days, interval)

if bench_series.empty:
    st.error(f"Benchmark data empty for {benchmark_key}.")
    st.stop()

def calculate_rrg(price: pd.Series, benchmark: pd.Series, window: int):
    aligned = pd.concat([price.rename("p"), benchmark.rename("b")], axis=1).dropna()
    if aligned.shape[0] < (2*window + 5):
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    rs = 100.0 * (aligned["p"] / aligned["b"])
    rs_mean = rs.rolling(window).mean()
    rs_std  = rs.rolling(window).std(ddof=0).replace(0, np.nan)

    rsr = 100.0 + (rs - rs_mean) / rs_std
    rsr = rsr.replace([np.inf, -np.inf], np.nan).dropna()
    if rsr.empty or rsr.shape[0] < (window + 3):
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    roc = rsr.pct_change(fill_method=None) * 100.0
    roc = roc.replace([np.inf, -np.inf], np.nan).dropna()
    if roc.empty or roc.shape[0] < (window + 3):
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    roc_mean = roc.rolling(window).mean()
    roc_std  = roc.rolling(window).std(ddof=0).replace(0, np.nan)

    rsm = 101.0 + (roc - roc_mean) / roc_std
    rsm = rsm.replace([np.inf, -np.inf], np.nan).dropna()

    idx = rsr.index.intersection(rsm.index)
    rsr, rsm = rsr.loc[idx], rsm.loc[idx]

    if rsr.empty or rsm.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    return rsr, rsm

def quadrant(x, y):
    if x < 100 and y < 100:
        return "Lagging"
    if x > 100 and y > 100:
        return "Leading"
    if x < 100 and y > 100:
        return "Improving"
    return "Weakening"

def q_color(q):
    return {"Lagging":"red","Leading":"green","Improving":"blue","Weakening":"orange"}.get(q,"gray")

rrg = {}
skipped = []

for k, s in sector_series_map.items():
    if s.empty:
        skipped.append((k, "No candle data"))
        continue
    rsr, rsm = calculate_rrg(s, bench_series, window)
    if rsr.empty or rsm.empty:
        skipped.append((k, "Insufficient/NaN after rolling"))
        continue
    rrg[k] = (rsr, rsm)

if not rrg:
    st.error("No sectors could be plotted. Increase days or reduce window.")
    st.write("Skipped:", skipped)
    st.stop()

base_len = min(len(v[0]) for v in rrg.values())
if base_len <= tail + 1:
    st.error("Not enough computed points. Increase days or reduce tail/window.")
    st.stop()

date_idx = st.slider("📅 RRG Time Index", tail, base_len-1, value=base_len-1, step=1)

fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.axhline(100, linestyle="--", color="black", linewidth=1)
ax.axvline(100, linestyle="--", color="black", linewidth=1)
ax.set_xlim(94, 106)
ax.set_ylim(94, 106)
ax.set_title(f"RRG vs {benchmark_key}")
ax.set_xlabel("JdK RS-Ratio")
ax.set_ylabel("JdK RS-Momentum")

ax.text(95, 105, "Improving")
ax.text(104, 105, "Leading")
ax.text(104, 95, "Weakening")
ax.text(95, 95, "Lagging")

rows = []
for name, (rsr, rsm) in rrg.items():
    x = rsr.iloc[date_idx - tail: date_idx + 1]
    y = rsm.iloc[date_idx - tail: date_idx + 1]
    q = quadrant(x.iloc[-1], y.iloc[-1])
    c = q_color(q)

    ax.plot(x.values, y.values, alpha=0.35, linewidth=1)
    ax.scatter(x.iloc[-1], y.iloc[-1], s=90, color=c)
    ax.text(x.iloc[-1], y.iloc[-1], name, fontsize=9)

    rows.append({
        "Index": name,
        "RS Ratio": round(float(x.iloc[-1]), 2),
        "RS Momentum": round(float(y.iloc[-1]), 2),
        "Quadrant": q,
        "Last Date": str(x.index[-1].date())
    })

c1, c2 = st.columns([1.2, 1])
with c1:
    st.pyplot(fig, use_container_width=True)
with c2:
    st.subheader("📋 Status")
    st.dataframe(pd.DataFrame(rows).sort_values(["Quadrant","Index"]), use_container_width=True)

if skipped:
    with st.expander("⚠️ Skipped"):
        st.dataframe(pd.DataFrame(skipped, columns=["Index","Reason"]), use_container_width=True)
