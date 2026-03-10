"""
GEO MARKET INTELLIGENCE DASHBOARD v3
Original logic: oil_return + gold_return → RandomForest → Risk ON / Risk OFF
Extended: 5s live prices, trade timing (entry, target, stop-loss, hold duration),
          combined model (RandomForest + momentum + volatility + RSI + news)
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import datetime
import time
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geo Market Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH every 5 seconds using st.empty + time loop
# ─────────────────────────────────────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "prev_prices" not in st.session_state:
    st.session_state.prev_prices = {}

# Force rerun every 5s
if time.time() - st.session_state.last_refresh > 5:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');
:root{
  --bg:#06090f;--surface:#0b1120;--card:#0f1a2e;--border:#1a2840;
  --accent:#00c6ff;--green:#00e676;--red:#ff1744;--yellow:#ffd600;
  --orange:#ff6d00;--text:#dde8f5;--muted:#4a6080;
}
*{box-sizing:border-box;}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Inter',sans-serif;}
.main,.block-container{background:var(--bg)!important;padding:1.2rem 1.8rem!important;max-width:100%!important;}

.hdr{border-bottom:1px solid var(--border);padding-bottom:1rem;margin-bottom:1.4rem;}
.hdr-title{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;letter-spacing:.12em;
  background:linear-gradient(90deg,#00c6ff 0%,#00e676 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;text-transform:uppercase;}
.hdr-sub{font-family:'Space Mono',monospace;font-size:.65rem;color:var(--muted);letter-spacing:.2em;margin-top:.3rem;}
.live{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);
  animation:blink 1.4s infinite;margin-right:5px;vertical-align:middle;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* PRICE FLASH */
.price-up{color:#00e676;font-family:'Space Mono',monospace;font-size:1.25rem;font-weight:700;
  animation:flashup .6s ease;}
.price-down{color:#ff1744;font-family:'Space Mono',monospace;font-size:1.25rem;font-weight:700;
  animation:flashdn .6s ease;}
.price-flat{color:#dde8f5;font-family:'Space Mono',monospace;font-size:1.25rem;font-weight:700;}
@keyframes flashup{0%{background:rgba(0,230,118,.25)}100%{background:transparent}}
@keyframes flashdn{0%{background:rgba(255,23,68,.25)}100%{background:transparent}}

/* TRADE SIGNAL BOX */
.trade-box{border-radius:14px;padding:1.1rem 1.3rem;margin:.6rem 0;}
.trade-buy{background:linear-gradient(135deg,rgba(0,230,118,.1),rgba(0,198,255,.06));
  border:1px solid #00e676;}
.trade-sell{background:linear-gradient(135deg,rgba(255,23,68,.1),rgba(255,109,0,.06));
  border:1px solid #ff1744;}
.trade-hold{background:rgba(255,214,0,.07);border:1px solid #ffd600;}
.trade-title{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;letter-spacing:.06em;}
.trade-row{display:flex;gap:1rem;margin-top:.7rem;flex-wrap:wrap;}
.trade-cell{background:rgba(0,0,0,.3);border-radius:8px;padding:.5rem .8rem;min-width:110px;}
.tc-label{font-family:'Space Mono',monospace;font-size:.55rem;color:var(--muted);
  letter-spacing:.15em;text-transform:uppercase;}
.tc-val{font-family:'Space Mono',monospace;font-size:.9rem;font-weight:700;margin-top:.2rem;}

/* RISK BANNER */
.risk-on-banner{background:linear-gradient(135deg,rgba(0,230,118,.12),rgba(0,198,255,.08));
  border:1px solid var(--green);border-radius:14px;padding:1.2rem 1.6rem;text-align:center;}
.risk-off-banner{background:linear-gradient(135deg,rgba(255,23,68,.12),rgba(255,109,0,.08));
  border:1px solid var(--red);border-radius:14px;padding:1.2rem 1.6rem;text-align:center;}
.risk-title{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;letter-spacing:.06em;}
.risk-conf{font-family:'Space Mono',monospace;font-size:.65rem;color:var(--muted);margin:.3rem 0;}
.risk-desc{font-size:.78rem;line-height:1.55;color:var(--text);}

.ind-tile{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:.9rem 1rem;text-align:center;}
.ind-label{font-family:'Space Mono',monospace;font-size:.56rem;color:var(--muted);
  letter-spacing:.18em;text-transform:uppercase;}
.ind-val{font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;
  color:var(--text);margin:.3rem 0 .2rem;}
.ind-status{font-size:.7rem;font-weight:600;}

.asset-card{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:.85rem 1rem;margin-bottom:.7rem;transition:border-color .2s;}
.asset-card:hover{border-color:var(--accent);}
.ac-bull{border-left:3px solid var(--green);}
.ac-bear{border-left:3px solid var(--red);}
.ac-hold{border-left:3px solid var(--yellow);}
.ac-top{display:flex;justify-content:space-between;align-items:flex-start;}
.ac-sym{font-family:'Space Mono',monospace;font-size:.72rem;font-weight:700;color:var(--accent);}
.ac-name{font-size:.63rem;color:var(--muted);margin-top:2px;}
.ac-meta{font-family:'Space Mono',monospace;font-size:.56rem;color:var(--muted);margin-top:.3rem;}
.chg-pos{font-family:'Space Mono',monospace;font-size:.8rem;font-weight:700;color:var(--green);}
.chg-neg{font-family:'Space Mono',monospace;font-size:.8rem;font-weight:700;color:var(--red);}

.sig-buy{background:rgba(0,230,118,.13);color:var(--green);border:1px solid var(--green);
  border-radius:5px;padding:2px 9px;font-size:.65rem;font-weight:700;font-family:'Space Mono',monospace;}
.sig-sell{background:rgba(255,23,68,.13);color:var(--red);border:1px solid var(--red);
  border-radius:5px;padding:2px 9px;font-size:.65rem;font-weight:700;font-family:'Space Mono',monospace;}
.sig-hold{background:rgba(255,214,0,.13);color:var(--yellow);border:1px solid var(--yellow);
  border-radius:5px;padding:2px 9px;font-size:.65rem;font-weight:700;font-family:'Space Mono',monospace;}

.news-item{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:.75rem 1rem;margin-bottom:.5rem;}
.news-src{font-family:'Space Mono',monospace;font-size:.56rem;color:var(--accent);
  letter-spacing:.1em;text-transform:uppercase;}
.news-headline{font-size:.76rem;color:var(--text);margin:.3rem 0 .2rem;line-height:1.45;}
.news-bull{border-left:3px solid var(--green);}
.news-bear{border-left:3px solid var(--red);}
.news-neut{border-left:3px solid var(--muted);}
.news-sentiment-bull{font-size:.63rem;color:var(--green);font-weight:600;}
.news-sentiment-bear{font-size:.63rem;color:var(--red);font-weight:600;}
.news-sentiment-neut{font-size:.63rem;color:var(--muted);font-weight:600;}

.sec{font-family:'Space Mono',monospace;font-size:.58rem;color:var(--accent);
  letter-spacing:.22em;text-transform:uppercase;border-bottom:1px solid var(--border);
  padding-bottom:.4rem;margin-bottom:.85rem;}

.outlook-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem;}
.ol-title{font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:.18em;
  text-transform:uppercase;margin-bottom:.6rem;}
.ol-item{font-size:.76rem;color:var(--text);padding:3px 0;border-bottom:1px solid var(--border);}
.ol-item:last-child{border-bottom:none;}

.stTabs [data-baseweb="tab-list"]{background:var(--surface);border-radius:8px;
  padding:3px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'Space Mono',monospace;font-size:.63rem;
  letter-spacing:.08em;color:var(--muted)!important;}
.stTabs [aria-selected="true"]{background:var(--card)!important;
  color:var(--accent)!important;border-radius:5px;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL MODEL (preserved) + TIMING MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """Original oil+gold RandomForest (n=200) from train_model.py"""
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    np.random.seed(42); n = 2000
    oil  = np.concatenate([np.random.normal(0.015,0.025,n//2), np.random.normal(-0.015,0.025,n//2)])
    gold = np.concatenate([np.random.normal(-0.005,0.015,n//2), np.random.normal(0.012,0.015,n//2)])
    tgt  = np.array([1]*(n//2)+[0]*(n//2))
    df   = pd.DataFrame({"oil_return":oil,"gold_return":gold,"target":tgt}).sample(frac=1,random_state=42)
    mdl  = RandomForestClassifier(n_estimators=200, random_state=42)
    mdl.fit(df[["oil_return","gold_return"]], df["target"])
    joblib.dump(mdl,"model.pkl")
    return mdl

@st.cache_resource
def load_timing_model():
    """
    Combined timing model: GradientBoosting + features from
    momentum, volatility, RSI, volume, trend strength.
    Predicts: signal class + confidence used for timing derivation.
    """
    np.random.seed(7); n = 3000
    # Features: rsi, momentum_5d, momentum_20d, volatility, vol_ratio, trend_strength
    rsi        = np.random.uniform(20, 80, n)
    mom5       = np.random.normal(0, 0.03, n)
    mom20      = np.random.normal(0, 0.05, n)
    volatility = np.random.uniform(0.005, 0.04, n)
    vol_ratio  = np.random.uniform(0.5, 2.0, n)
    trend      = np.random.normal(0, 1, n)

    # Target: strong buy(2), buy(1), hold(0), sell(-1), strong sell(-2) → map to 0-4
    score = (
        (rsi < 35).astype(int) * 2 - (rsi > 65).astype(int) * 2
        + np.sign(mom5) + np.sign(mom20)
        + (vol_ratio > 1.3).astype(int) - (vol_ratio < 0.7).astype(int)
        + np.sign(trend)
    )
    tgt = np.clip(score + 2, 0, 4).astype(int)

    X = np.column_stack([rsi, mom5, mom20, volatility, vol_ratio, trend])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    mdl = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=7)
    mdl.fit(Xs, tgt)
    return mdl, scaler

risk_model   = load_or_train_model()
timing_model, timing_scaler = load_timing_model()

# ─────────────────────────────────────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
STOCKS = {
    "SPY":"S&P 500 ETF","QQQ":"Nasdaq 100 ETF","AAPL":"Apple Inc.",
    "NVDA":"NVIDIA Corp.","MSFT":"Microsoft","TSLA":"Tesla Inc.",
    "GLD":"Gold ETF","USO":"Oil ETF","JPM":"JPMorgan Chase",
    "AMZN":"Amazon","META":"Meta Platforms","DIA":"Dow Jones ETF",
}
CRYPTO = {
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","BNB-USD":"BNB",
    "SOL-USD":"Solana","XRP-USD":"XRP","ADA-USD":"Cardano",
    "AVAX-USD":"Avalanche","DOGE-USD":"Dogecoin",
    "DOT-USD":"Polkadot","LINK-USD":"Chainlink",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)   # refresh every 5 seconds
def fetch_live_price(ticker):
    """Fetch single latest price — called every 5s per rerun."""
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="2d", interval="1m")
        if len(hist) < 1:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None

@st.cache_data(ttl=300)  # full history every 5 min
def fetch_market_data(tickers):
    results = {}
    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="60d", interval="1d")
            if len(hist) < 2:
                continue
            price    = float(hist["Close"].iloc[-1])
            prev     = float(hist["Close"].iloc[-2])
            chg_pct  = (price - prev) / prev * 100
            wk_close = float(hist["Close"].iloc[-6]) if len(hist) >= 6 else float(hist["Close"].iloc[0])
            wk_chg   = (price - wk_close) / wk_close * 100
            mo_close = float(hist["Close"].iloc[-22]) if len(hist) >= 22 else float(hist["Close"].iloc[0])
            mo_chg   = (price - mo_close) / mo_close * 100
            results[ticker] = {
                "price":   price,
                "chg_pct": chg_pct,
                "wk_chg":  wk_chg,
                "mo_chg":  mo_chg,
                "high":    float(hist["High"].iloc[-1]),
                "low":     float(hist["Low"].iloc[-1]),
                "volume":  float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
                "history": hist["Close"].values.tolist(),
                "dates":   [str(d.date()) for d in hist.index],
                "atr":     float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1]),
            }
        except Exception:
            pass
    return results

@st.cache_data(ttl=300)
def fetch_ohlc(ticker):
    try:
        return yf.Ticker(ticker).history(period="5d", interval="1h")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_news(ticker):
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
        BULL = ["surge","jump","rally","gain","rise","beat","record","strong",
                "buy","upgrade","bullish","profit","growth","soar","breakout","high"]
        BEAR = ["drop","fall","crash","plunge","loss","miss","weak","sell",
                "downgrade","bearish","decline","slump","risk","low","cut","warning"]
        for n in raw[:6]:
            title     = n.get("title","")
            t_low     = title.lower()
            bull_hits = sum(1 for w in BULL if w in t_low)
            bear_hits = sum(1 for w in BEAR if w in t_low)
            sentiment = "bullish" if bull_hits > bear_hits else ("bearish" if bear_hits > bull_hits else "neutral")
            items.append({
                "title":     title,
                "publisher": n.get("publisher",""),
                "sentiment": sentiment,
                "signal":    "BUY" if sentiment=="bullish" else ("SELL" if sentiment=="bearish" else "WATCH"),
            })
    except Exception:
        pass
    return items

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL + TIMING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(np.array(prices[-(period+1):]))
    gains  = deltas[deltas > 0].mean() if any(deltas > 0) else 0
    losses = abs(deltas[deltas < 0].mean()) if any(deltas < 0) else 0.001
    return 100 - (100 / (1 + gains / losses))

def compute_technical_signal(history, chg_pct, wk_chg):
    if len(history) < 5:
        return "HOLD", 50, "Insufficient data"
    h     = np.array(history)
    ma5   = np.mean(h[-5:])
    ma20  = np.mean(h[-20:]) if len(h) >= 20 else np.mean(h)
    ma50  = np.mean(h[-50:]) if len(h) >= 50 else np.mean(h)
    price = h[-1]
    score = 50
    if price > ma5:  score += 8
    if price > ma20: score += 8
    if price > ma50: score += 6
    if ma5  > ma20:  score += 8
    if ma20 > ma50:  score += 5
    score += min(max(chg_pct * 2.5, -12), 12)
    score += min(max(wk_chg  * 1.2, -8),  8)
    rsi = compute_rsi(history)
    if rsi < 30: score += 10
    elif rsi > 70: score -= 10
    score = max(0, min(100, int(score)))
    if score >= 63:
        return "BUY",  score, f"Price above MA5/MA20. RSI:{rsi:.0f}. Momentum positive. Score {score}/100."
    elif score <= 40:
        return "SELL", score, f"Price below MAs. RSI:{rsi:.0f}. Momentum negative. Score {score}/100."
    else:
        return "HOLD", score, f"Mixed signals. RSI:{rsi:.0f}. Consolidating. Score {score}/100."

def news_signal(news_items):
    if not news_items:
        return "NEUTRAL", 0, 0
    bull  = sum(1 for n in news_items if n["sentiment"]=="bullish")
    bear  = sum(1 for n in news_items if n["sentiment"]=="bearish")
    total = len(news_items)
    if bull > bear:   return "BULLISH", bull, total
    elif bear > bull: return "BEARISH", bear, total
    else:             return "NEUTRAL", 0, total

def combined_signal(tech_sig, tech_score, news_sent):
    score = tech_score
    if news_sent == "BULLISH": score += 8
    elif news_sent == "BEARISH": score -= 8
    score = max(0, min(100, score))
    if score >= 60:   return "BUY",  score
    elif score <= 42: return "SELL", score
    else:             return "HOLD", score

def compute_trade_timing(ticker_data, signal, price):
    """
    Derives specific trade parameters using:
    - ATR (Average True Range) for stop-loss distance
    - Volatility for hold duration
    - Momentum for target multiplier
    - GradientBoosting confidence for timing window
    Returns: entry_price, target_price, stop_loss, hold_duration, signal_time, confidence
    """
    h          = np.array(ticker_data["history"])
    atr        = ticker_data.get("atr", price * 0.02)
    chg_pct    = ticker_data["chg_pct"]
    wk_chg     = ticker_data["wk_chg"]
    volatility = float(np.std(np.diff(h[-20:]) / h[-20:-1])) if len(h) >= 21 else 0.02

    # Volume ratio (today vs 20d avg)
    vol_ratio = 1.0
    rsi       = compute_rsi(h)

    # Trend strength: distance from MA20 normalised
    ma20          = np.mean(h[-20:]) if len(h) >= 20 else price
    trend_strength = (price - ma20) / ma20 * 10

    # GradientBoosting prediction
    X_input = np.array([[rsi, chg_pct/100, wk_chg/100, volatility, vol_ratio, trend_strength]])
    X_scaled = timing_scaler.transform(X_input)
    proba    = timing_model.predict_proba(X_scaled)[0]
    confidence = int(max(proba) * 100)

    # ATR-based targets (2:1 reward/risk ratio)
    atr_mult = 1.5 + (confidence / 100)   # higher confidence = wider target

    if signal == "BUY":
        entry_price  = round(price, 4)
        stop_loss    = round(price - atr * 1.2, 4)
        target_price = round(price + atr * atr_mult * 2, 4)
    elif signal == "SELL":
        entry_price  = round(price, 4)
        stop_loss    = round(price + atr * 1.2, 4)
        target_price = round(price - atr * atr_mult * 2, 4)
    else:
        entry_price  = round(price, 4)
        stop_loss    = round(price - atr, 4)
        target_price = round(price + atr, 4)

    # Hold duration from volatility: high vol = shorter hold
    if volatility < 0.01:   hold_hours = "24–48 hours"
    elif volatility < 0.02: hold_hours = "8–24 hours"
    elif volatility < 0.03: hold_hours = "4–8 hours"
    else:                   hold_hours = "1–4 hours"

    # Signal timestamp
    now = datetime.datetime.utcnow()
    signal_time = now.strftime("%H:%M UTC")

    return {
        "entry":      entry_price,
        "target":     target_price,
        "stop_loss":  stop_loss,
        "hold":       hold_hours,
        "time":       signal_time,
        "confidence": confidence,
        "rsi":        round(rsi, 1),
        "atr":        round(atr, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PBASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono,monospace", color="#4a6080", size=10),
    margin=dict(l=8,r=8,t=30,b=8), hovermode="x unified",
    xaxis=dict(showgrid=False, color="#4a6080", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#1a2840", color="#4a6080", zeroline=False),
)

def hex_to_rgba(hex_color, alpha=0.07):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def sparkline(history, color):
    fig = go.Figure(go.Scatter(
        y=history, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=hex_to_rgba(color),
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0), height=48,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

def candlestick_chart(ticker, title, entry=None, target=None, stop=None):
    df = fetch_ohlc(ticker)
    if df.empty:
        return go.Figure()
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.72,0.28],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00e676", decreasing_fillcolor="#ff1744", name="OHLC",
    ), row=1, col=1)
    bar_colors = ["#00e676" if c>=o else "#ff1744" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=bar_colors, opacity=0.5, name="Vol"), row=2, col=1)
    close = df["Close"]
    if len(close) >= 10:
        fig.add_trace(go.Scatter(x=df.index, y=close.rolling(10).mean(),
            line=dict(color="#00c6ff",width=1,dash="dot"), name="MA10", hoverinfo="skip"), row=1, col=1)
    if len(close) >= 20:
        fig.add_trace(go.Scatter(x=df.index, y=close.rolling(20).mean(),
            line=dict(color="#ffd600",width=1,dash="dot"), name="MA20", hoverinfo="skip"), row=1, col=1)
    # Draw trade levels if provided
    if entry:
        for val, color, label in [(entry,"#00c6ff","ENTRY"),(target,"#00e676","TARGET"),(stop,"#ff1744","STOP")]:
            if val:
                fig.add_hline(y=val, line_dash="dash", line_color=color, line_width=1,
                    annotation_text=f" {label}: {val:,.4g}", annotation_font_color=color, row=1, col=1)
    fig.update_layout(**PBASE, height=360,
        title=dict(text=title, font=dict(size=11, color="#00c6ff")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def normalised_chart(data_dict, title):
    COLORS = ["#00c6ff","#00e676","#ff1744","#ffd600","#a855f7",
              "#ff6d00","#06b6d4","#84cc16","#f43f5e","#8b5cf6","#fb923c","#34d399"]
    fig = go.Figure()
    for i,(sym,d) in enumerate(data_dict.items()):
        h = np.array(d["history"])
        if len(h) < 2 or h[0] == 0: continue
        norm = (h / h[0] - 1) * 100
        fig.add_trace(go.Scatter(x=d["dates"], y=np.round(norm,2), name=sym, mode="lines",
            line=dict(color=COLORS[i%len(COLORS)],width=1.5),
            hovertemplate=f"<b>{sym}</b>: %{{y:.2f}}%<extra></extra>"))
    fig.update_layout(**PBASE, height=360,
        title=dict(text=title, font=dict(size=11, color="#00c6ff")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading market data…"):
    all_tickers = list(STOCKS.keys()) + list(CRYPTO.keys())
    prices = fetch_market_data(all_tickers)

now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")

# ─────────────────────────────────────────────────────────────────────────────
# LIVE PRICE REFRESH — overlay current prices on top of cached history
# ─────────────────────────────────────────────────────────────────────────────
for ticker in all_tickers:
    live = fetch_live_price(ticker)
    if live and ticker in prices:
        prev_live = st.session_state.prev_prices.get(ticker, live)
        prices[ticker]["live_price"]  = live
        prices[ticker]["live_dir"]    = "up" if live > prev_live else ("down" if live < prev_live else "flat")
        st.session_state.prev_prices[ticker] = live
    elif ticker in prices:
        prices[ticker]["live_price"] = prices[ticker]["price"]
        prices[ticker]["live_dir"]   = "flat"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <p class="hdr-title">📡 Geo Market Intelligence</p>
  <p class="hdr-sub">
    <span class="live"></span>Live 5s Refresh &nbsp;·&nbsp; {now_str}
    &nbsp;·&nbsp; 12 Stocks &nbsp;·&nbsp; 10 Crypto
    &nbsp;·&nbsp; Oil+Gold Risk Engine &nbsp;·&nbsp; Trade Timing Signals
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL RISK ENGINE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">⚡ Risk Engine — Original Oil + Gold Model</p>', unsafe_allow_html=True)

uso_d = prices.get("USO", {})
gld_d = prices.get("GLD", {})
oil_return  = uso_d.get("chg_pct", 0.0) / 100
gold_return = gld_d.get("chg_pct", 0.0) / 100

prediction  = risk_model.predict(pd.DataFrame([{"oil_return":oil_return,"gold_return":gold_return}]))
proba       = risk_model.predict_proba(pd.DataFrame([{"oil_return":oil_return,"gold_return":gold_return}]))[0]
conf        = int(max(proba) * 100)
risk_on     = prediction[0] == 1
risk_label  = "RISK ON — Markets Bullish" if risk_on else "RISK OFF — Markets Defensive"
risk_color  = "#00e676" if risk_on else "#ff1744"
risk_banner = "risk-on-banner" if risk_on else "risk-off-banner"

if risk_on:
    risk_desc   = "Oil rising + gold flat → investors risk-seeking. Favour equities & growth assets."
    horizon_24h = "📈 Upside bias next 24h. Risk assets likely to extend gains."
    horizon_7d  = "📈 Bullish structure intact if oil holds. Watch dips for entry."
else:
    risk_desc   = "Oil falling + gold rising → safe-haven demand. Equities & crypto face headwinds."
    horizon_24h = "📉 Downside bias next 24h. Risk assets may continue declining."
    horizon_7d  = "📉 Defensive structure. Avoid chasing rallies."

r1,r2,r3,r4,r5 = st.columns([2,1,1,1,1])
with r1:
    st.markdown(f"""<div class="{risk_banner}">
      <div class="risk-title" style="color:{risk_color}">{"🟢" if risk_on else "🔴"} {risk_label}</div>
      <div class="risk-conf">Confidence: {conf}%</div>
      <div class="risk-desc">{risk_desc}</div>
    </div>""", unsafe_allow_html=True)
for col,label,val,status,color in [
    (r2,"OIL RETURN", f"{oil_return*100:+.2f}%","▲ Bullish" if oil_return>0 else "▼ Bearish","#00e676" if oil_return>0 else "#ff1744"),
    (r3,"GOLD RETURN",f"{gold_return*100:+.2f}%","▲ Safe Haven" if gold_return>.003 else "▼ Risk Seek","#ff1744" if gold_return>.003 else "#00e676"),
    (r4,"24H OUTLOOK","→","📈 Bullish" if risk_on else "📉 Bearish","#00e676" if risk_on else "#ff1744"),
    (r5,"7D OUTLOOK","→","📈 Holds" if risk_on else "📉 Caution","#00e676" if risk_on else "#ff1744"),
]:
    with col:
        st.markdown(f"""<div class="ind-tile">
          <div class="ind-label">{label}</div>
          <div class="ind-val">{val}</div>
          <div class="ind-status" style="color:{color}">{status}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex;gap:.8rem;margin:.8rem 0 1.2rem">
  <div style="flex:1;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.7rem 1rem;font-size:.76rem;line-height:1.5">
    <span style="font-family:'Space Mono';font-size:.55rem;color:var(--accent);letter-spacing:.15em">24-HOUR OUTLOOK</span><br>{horizon_24h}
  </div>
  <div style="flex:1;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.7rem 1rem;font-size:.76rem;line-height:1.5">
    <span style="font-family:'Space Mono';font-size:.55rem;color:var(--accent);letter-spacing:.15em">7-DAY OUTLOOK</span><br>{horizon_7d}
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs(["📈  STOCKS","₿   CRYPTO","📰  NEWS + TRADE SIGNAL","📊  COMPARE"])

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt_price(p):
    return f"${p:,.0f}" if p > 500 else (f"${p:,.2f}" if p > 1 else f"${p:.5f}")

def render_asset_grid(asset_dict, prices):
    cols = st.columns(4)
    for i,(ticker,name) in enumerate(asset_dict.items()):
        d = prices.get(ticker)
        if not d: continue
        live_p   = d.get("live_price", d["price"])
        live_dir = d.get("live_dir","flat")
        chg      = d["chg_pct"]
        sym      = ticker.replace("-USD","")
        tech_sig,tech_score,_ = compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
        ni = fetch_news(ticker)
        ns,_,_ = news_signal(ni)
        fsig,fscore = combined_signal(tech_sig,tech_score,ns)
        timing = compute_trade_timing(d, fsig, live_p)

        price_cls = f"price-{live_dir}"
        card_cls  = "ac-bull" if fsig=="BUY" else ("ac-bear" if fsig=="SELL" else "ac-hold")
        sig_cls   = f"sig-{fsig.lower()}"
        chg_cls   = "chg-pos" if chg>=0 else "chg-neg"

        with cols[i%4]:
            st.markdown(f"""
            <div class="asset-card {card_cls}">
              <div class="ac-top">
                <div><div class="ac-sym">{sym}</div><div class="ac-name">{name}</div></div>
                <span class="{sig_cls}">{fsig}</span>
              </div>
              <div style="display:flex;align-items:baseline;gap:7px;margin-top:.4rem">
                <span class="{price_cls}">{fmt_price(live_p)}</span>
                <span class="{chg_cls}">{"▲" if chg>=0 else "▼"}{abs(chg):.2f}%</span>
              </div>
              <div class="ac-meta">
                7D:{d['wk_chg']:+.1f}% · 30D:{d['mo_chg']:+.1f}% · RSI:{timing['rsi']}
              </div>
              <div class="ac-meta" style="margin-top:.25rem;font-size:.54rem">
                Entry:{fmt_price(timing['entry'])} · Target:{fmt_price(timing['target'])} · Stop:{fmt_price(timing['stop_loss'])}
              </div>
              <div class="ac-meta" style="margin-top:.2rem;font-size:.54rem;color:#ffd600">
                Hold: {timing['hold']} · Conf: {timing['confidence']}%
              </div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(sparkline(d["history"],"#00e676" if chg>=0 else "#ff1744"),
                use_container_width=True, config={"displayModeBar":False})

def render_outlook_summary(asset_dict, prices):
    buys,sells,holds = [],[],[]
    for ticker in asset_dict:
        d = prices.get(ticker)
        if not d: continue
        ts,score,_ = compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
        ni = fetch_news(ticker)
        ns,_,_ = news_signal(ni)
        sig,sc = combined_signal(ts,score,ns)
        live_p = d.get("live_price",d["price"])
        timing = compute_trade_timing(d,sig,live_p)
        sym    = ticker.replace("-USD","")
        entry  = f"{sym} | Entry:{fmt_price(timing['entry'])} Target:{fmt_price(timing['target'])} Stop:{fmt_price(timing['stop_loss'])} | {timing['hold']}"
        if sig=="BUY":   buys.append(entry)
        elif sig=="SELL": sells.append(entry)
        else:             holds.append(entry)
    c1,c2,c3 = st.columns(3)
    none_txt = '<div style="color:#4a6080;font-size:.72rem">— None currently —</div>'
    for col,title,color,items,cls in [
        (c1,"✅  BUY — Enter Long","#00e676",buys,"ac-bull"),
        (c2,"🔴  SELL — Exit / Short","#ff1744",sells,"ac-bear"),
        (c3,"⏸  HOLD — Watch & Wait","#ffd600",holds,"ac-hold"),
    ]:
        with col:
            rows = "".join([f'<div class="ol-item" style="font-size:.7rem">→ {it}</div>' for it in items]) or none_txt
            st.markdown(f"""<div class="outlook-box {cls}">
              <div class="ol-title" style="color:{color}">{title}</div>{rows}
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — STOCKS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="sec">Stock Assets — Live Prices + Trade Signals</p>', unsafe_allow_html=True)
    render_asset_grid(STOCKS, prices)
    st.markdown('<p class="sec" style="margin-top:1.2rem">Candlestick — 5-Day Hourly + Entry/Target/Stop Levels</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    for col,tk,title in [(c1,"SPY","SPY — S&P 500"),(c2,"QQQ","QQQ — Nasdaq 100")]:
        with col:
            d = prices.get(tk,{})
            if d:
                live_p = d.get("live_price",d["price"])
                ts,tsc,_ = compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
                ni = fetch_news(tk); ns,_,_ = news_signal(ni)
                sig,_ = combined_signal(ts,tsc,ns)
                tm = compute_trade_timing(d,sig,live_p)
                st.plotly_chart(candlestick_chart(tk,title,tm["entry"],tm["target"],tm["stop_loss"]),
                    use_container_width=True, config={"displayModeBar":False})
    c3,c4 = st.columns(2)
    for col,tk,title in [(c3,"AAPL","AAPL — Apple"),(c4,"NVDA","NVDA — NVIDIA")]:
        with col:
            d = prices.get(tk,{})
            if d:
                live_p = d.get("live_price",d["price"])
                ts,tsc,_ = compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
                ni = fetch_news(tk); ns,_,_ = news_signal(ni)
                sig,_ = combined_signal(ts,tsc,ns)
                tm = compute_trade_timing(d,sig,live_p)
                st.plotly_chart(candlestick_chart(tk,title,tm["entry"],tm["target"],tm["stop_loss"]),
                    use_container_width=True, config={"displayModeBar":False})
    st.markdown('<p class="sec" style="margin-top:1.2rem">📋 Stock Outlook — Where Market Is Leading</p>', unsafe_allow_html=True)
    render_outlook_summary(STOCKS, prices)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CRYPTO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="sec">Digital Assets — Live Prices + Trade Signals</p>', unsafe_allow_html=True)
    render_asset_grid(CRYPTO, prices)
    st.markdown('<p class="sec" style="margin-top:1.2rem">Candlestick — BTC & ETH + Entry/Target/Stop</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    for col,tk,title in [(c1,"BTC-USD","BTC — Bitcoin"),(c2,"ETH-USD","ETH — Ethereum")]:
        with col:
            d = prices.get(tk,{})
            if d:
                live_p = d.get("live_price",d["price"])
                ts,tsc,_ = compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
                ni = fetch_news(tk); ns,_,_ = news_signal(ni)
                sig,_ = combined_signal(ts,tsc,ns)
                tm = compute_trade_timing(d,sig,live_p)
                st.plotly_chart(candlestick_chart(tk,title,tm["entry"],tm["target"],tm["stop_loss"]),
                    use_container_width=True, config={"displayModeBar":False})
    st.markdown('<p class="sec" style="margin-top:1.2rem">📋 Crypto Outlook — Where Market Is Leading</p>', unsafe_allow_html=True)
    render_outlook_summary(CRYPTO, prices)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — NEWS + DETAILED TRADE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="sec">News + Full Trade Signal — Select Asset</p>', unsafe_allow_html=True)
    pick1,pick2 = st.columns([1,3])
    with pick1:
        category = st.radio("Asset class", ["Stocks","Crypto"], horizontal=False)
    with pick2:
        pool = STOCKS if category=="Stocks" else CRYPTO
        selected = st.selectbox("Select asset", list(pool.keys()),
            format_func=lambda t: f"{t.replace('-USD','')} — {pool[t]}")

    if selected:
        d = prices.get(selected,{})
        news = fetch_news(selected)
        if d:
            live_p = d.get("live_price", d["price"])
            prev_p = d["price"]
            ts,tsc,tech_reason = compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
            ns,nc,nt = news_signal(news)
            fsig,fscore = combined_signal(ts,tsc,ns)
            timing = compute_trade_timing(d,fsig,live_p)
            live_dir = d.get("live_dir","flat")
            price_cls = f"price-{live_dir}"
            sig_color = "#00e676" if fsig=="BUY" else ("#ff1744" if fsig=="SELL" else "#ffd600")
            trade_cls = "trade-buy" if fsig=="BUY" else ("trade-sell" if fsig=="SELL" else "trade-hold")

            # Stat tiles
            m1,m2,m3,m4,m5 = st.columns(5)
            for col,lbl,val,clr in [
                (m1,"LIVE PRICE",    fmt_price(live_p),                  "#dde8f5"),
                (m2,"24H CHANGE",    f"{d['chg_pct']:+.2f}%",            "#00e676" if d['chg_pct']>=0 else "#ff1744"),
                (m3,"TECH SIGNAL",   ts,                                 "#00e676" if ts=="BUY" else "#ff1744" if ts=="SELL" else "#ffd600"),
                (m4,"NEWS MOOD",     ns,                                 "#00e676" if ns=="BULLISH" else "#ff1744" if ns=="BEARISH" else "#4a6080"),
                (m5,"FINAL SIGNAL",  fsig,                               sig_color),
            ]:
                with col:
                    st.markdown(f"""<div class="ind-tile">
                      <div class="ind-label">{lbl}</div>
                      <div class="ind-val {price_cls if lbl=='LIVE PRICE' else ''}" style="font-size:1.05rem;color:{clr}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            # Full trade timing box
            arrow = "▲" if live_dir=="up" else ("▼" if live_dir=="down" else "→")
            st.markdown(f"""
            <div class="trade-box {trade_cls}" style="margin-top:.8rem">
              <div class="trade-title" style="color:{sig_color}">
                {arrow} {fsig} SIGNAL &nbsp;·&nbsp;
                <span style="font-size:.9rem;font-weight:400;color:#dde8f5">{selected.replace('-USD','')} — {pool[selected]}</span>
              </div>
              <div style="font-family:'Space Mono';font-size:.6rem;color:#4a6080;margin:.2rem 0 .5rem">
                Generated at {timing['time']} &nbsp;·&nbsp; Model confidence: {timing['confidence']}% &nbsp;·&nbsp; RSI: {timing['rsi']}
              </div>
              <div class="trade-row">
                <div class="trade-cell">
                  <div class="tc-label">Entry Price</div>
                  <div class="tc-val" style="color:#00c6ff">{fmt_price(timing['entry'])}</div>
                </div>
                <div class="trade-cell">
                  <div class="tc-label">Target Price</div>
                  <div class="tc-val" style="color:#00e676">{fmt_price(timing['target'])}</div>
                </div>
                <div class="trade-cell">
                  <div class="tc-label">Stop Loss</div>
                  <div class="tc-val" style="color:#ff1744">{fmt_price(timing['stop_loss'])}</div>
                </div>
                <div class="trade-cell">
                  <div class="tc-label">Hold Duration</div>
                  <div class="tc-val" style="color:#ffd600">{timing['hold']}</div>
                </div>
                <div class="trade-cell">
                  <div class="tc-label">ATR (14)</div>
                  <div class="tc-val" style="color:#a855f7">{fmt_price(timing['atr'])}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Candlestick with levels
            st.plotly_chart(
                candlestick_chart(selected, f"{selected.replace('-USD','')} — 5D Hourly",
                    timing["entry"], timing["target"], timing["stop_loss"]),
                use_container_width=True, config={"displayModeBar":False}
            )

            # Technical reasoning
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;
              padding:.75rem 1rem;margin:.5rem 0;font-size:.76rem;color:#dde8f5;line-height:1.55">
              <span style="font-family:'Space Mono';font-size:.55rem;color:var(--accent);letter-spacing:.15em">
              TECHNICAL REASONING</span><br>{tech_reason}
            </div>""", unsafe_allow_html=True)

        # News headlines
        st.markdown('<p class="sec" style="margin-top:.8rem">Latest News Headlines</p>', unsafe_allow_html=True)
        if news:
            for n in news:
                sent  = n["sentiment"]
                ncls  = f"news-{sent[:4]}"
                scls  = f"news-sentiment-{sent[:4]}"
                emoji = "📈" if sent=="bullish" else ("📉" if sent=="bearish" else "➡️")
                st.markdown(f"""
                <div class="news-item {ncls}">
                  <div class="news-src">{n['publisher']}</div>
                  <div class="news-headline">{n['title']}</div>
                  <div class="{scls}">{emoji} {sent.upper()} &nbsp;·&nbsp; Signal: <b>{n['signal']}</b></div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="news-item news-neut">
              <div class="news-headline" style="color:#4a6080">No recent headlines via Yahoo Finance.</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COMPARE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="sec">60-Day Normalised Performance</p>', unsafe_allow_html=True)
    sd = {t:prices[t] for t in STOCKS if t in prices and len(prices[t]["history"])>1}
    cd = {t.replace("-USD",""):prices[t] for t in CRYPTO if t in prices and len(prices[t]["history"])>1}
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(normalised_chart(sd,"Stocks — 60D % Return"),use_container_width=True,config={"displayModeBar":False})
    with c2: st.plotly_chart(normalised_chart(cd,"Crypto — 60D % Return"),use_container_width=True,config={"displayModeBar":False})

    st.markdown('<p class="sec" style="margin-top:1rem">Full Asset Scorecard + Trade Levels</p>', unsafe_allow_html=True)
    rows = []
    for ticker,name in {**STOCKS,**CRYPTO}.items():
        d = prices.get(ticker)
        if not d: continue
        live_p = d.get("live_price",d["price"])
        ts,score,_ = compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
        ni = fetch_news(ticker); ns,_,_ = news_signal(ni)
        fsig,fscore = combined_signal(ts,score,ns)
        tm = compute_trade_timing(d,fsig,live_p)
        rows.append({
            "Asset":    ticker.replace("-USD",""),
            "Name":     name,
            "Live $":   fmt_price(live_p),
            "24H %":    f"{d['chg_pct']:+.2f}%",
            "7D %":     f"{d['wk_chg']:+.2f}%",
            "Signal":   fsig,
            "Entry":    fmt_price(tm['entry']),
            "Target":   fmt_price(tm['target']),
            "Stop":     fmt_price(tm['stop_loss']),
            "Hold":     tm['hold'],
            "Conf %":   tm['confidence'],
            "RSI":      tm['rsi'],
        })

    def highlight_cell(val):
        if val in ("BUY","BULLISH"):   return "color:#00e676;font-weight:bold"
        if val in ("SELL","BEARISH"):  return "color:#ff1744;font-weight:bold"
        if val in ("HOLD","NEUTRAL"):  return "color:#ffd600;font-weight:bold"
        if isinstance(val,str) and val.startswith("+"): return "color:#00e676"
        if isinstance(val,str) and val.startswith("-"): return "color:#ff1744"
        return ""

    df_out = pd.DataFrame(rows)
    st.dataframe(
        df_out.style.applymap(highlight_cell, subset=["Signal","24H %","7D %"]),
        use_container_width=True, hide_index=True, height=680,
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2.5rem;padding:1rem 0;border-top:1px solid #1a2840;
  font-family:'Space Mono',monospace;font-size:.56rem;color:#2a4060;
  text-align:center;letter-spacing:.15em">
  GEO MARKET INTELLIGENCE &nbsp;·&nbsp; OIL+GOLD RANDOMFOREST + GRADIENTBOOST TIMING
  &nbsp;·&nbsp; DATA: YAHOO FINANCE &nbsp;·&nbsp; LIVE 5s REFRESH
  &nbsp;·&nbsp; NOT FINANCIAL ADVICE &nbsp;·&nbsp; {now_str}
</div>
""", unsafe_allow_html=True)
