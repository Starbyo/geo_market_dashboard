"""
GEO MARKET INTELLIGENCE DASHBOARD
Original logic: oil_return + gold_return → RandomForest → Risk ON / Risk OFF
Extended with: live prices, news headlines, technical signals, buy/sell per asset
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

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

/* HEADER */
.hdr{border-bottom:1px solid var(--border);padding-bottom:1rem;margin-bottom:1.4rem;}
.hdr-title{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;letter-spacing:.12em;
  background:linear-gradient(90deg,#00c6ff 0%,#00e676 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;text-transform:uppercase;}
.hdr-sub{font-family:'Space Mono',monospace;font-size:.65rem;color:var(--muted);letter-spacing:.2em;margin-top:.3rem;}
.live{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);
  animation:blink 1.4s infinite;margin-right:5px;vertical-align:middle;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* RISK BANNER */
.risk-on-banner{background:linear-gradient(135deg,rgba(0,230,118,.12),rgba(0,198,255,.08));
  border:1px solid var(--green);border-radius:14px;padding:1.2rem 1.6rem;text-align:center;}
.risk-off-banner{background:linear-gradient(135deg,rgba(255,23,68,.12),rgba(255,109,0,.08));
  border:1px solid var(--red);border-radius:14px;padding:1.2rem 1.6rem;text-align:center;}
.risk-title{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;letter-spacing:.08em;}
.risk-conf{font-family:'Space Mono',monospace;font-size:.68rem;color:var(--muted);margin:.3rem 0;}
.risk-desc{font-size:.8rem;line-height:1.55;color:var(--text);}

/* INDICATOR TILE */
.ind-tile{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:.9rem 1rem;text-align:center;}
.ind-label{font-family:'Space Mono',monospace;font-size:.58rem;color:var(--muted);
  letter-spacing:.18em;text-transform:uppercase;}
.ind-val{font-family:'Space Mono',monospace;font-size:1.4rem;font-weight:700;
  color:var(--text);margin:.35rem 0 .25rem;}
.ind-status{font-size:.72rem;font-weight:600;}

/* ASSET CARD */
.asset-card{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:.85rem 1rem;margin-bottom:.7rem;transition:border-color .2s;}
.asset-card:hover{border-color:var(--accent);}
.ac-bull{border-left:3px solid var(--green);}
.ac-bear{border-left:3px solid var(--red);}
.ac-hold{border-left:3px solid var(--yellow);}
.ac-top{display:flex;justify-content:space-between;align-items:flex-start;}
.ac-sym{font-family:'Space Mono',monospace;font-size:.72rem;font-weight:700;color:var(--accent);}
.ac-name{font-size:.65rem;color:var(--muted);margin-top:2px;}
.ac-price{font-family:'Space Mono',monospace;font-size:1.2rem;font-weight:700;color:var(--text);}
.ac-chg-pos{font-family:'Space Mono',monospace;font-size:.8rem;font-weight:700;color:var(--green);}
.ac-chg-neg{font-family:'Space Mono',monospace;font-size:.8rem;font-weight:700;color:var(--red);}
.ac-meta{font-family:'Space Mono',monospace;font-size:.58rem;color:var(--muted);margin-top:.35rem;}

/* SIGNAL BADGE */
.sig-buy{background:rgba(0,230,118,.13);color:var(--green);border:1px solid var(--green);
  border-radius:5px;padding:2px 9px;font-size:.65rem;font-weight:700;font-family:'Space Mono',monospace;}
.sig-sell{background:rgba(255,23,68,.13);color:var(--red);border:1px solid var(--red);
  border-radius:5px;padding:2px 9px;font-size:.65rem;font-weight:700;font-family:'Space Mono',monospace;}
.sig-hold{background:rgba(255,214,0,.13);color:var(--yellow);border:1px solid var(--yellow);
  border-radius:5px;padding:2px 9px;font-size:.65rem;font-weight:700;font-family:'Space Mono',monospace;}

/* NEWS */
.news-item{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:.75rem 1rem;margin-bottom:.5rem;}
.news-src{font-family:'Space Mono',monospace;font-size:.58rem;color:var(--accent);
  letter-spacing:.1em;text-transform:uppercase;}
.news-headline{font-size:.78rem;color:var(--text);margin:.3rem 0 .2rem;line-height:1.45;}
.news-bull{border-left:3px solid var(--green);}
.news-bear{border-left:3px solid var(--red);}
.news-neut{border-left:3px solid var(--muted);}
.news-sentiment-bull{font-size:.65rem;color:var(--green);font-weight:600;}
.news-sentiment-bear{font-size:.65rem;color:var(--red);font-weight:600;}
.news-sentiment-neut{font-size:.65rem;color:var(--muted);font-weight:600;}

/* SECTION */
.sec{font-family:'Space Mono',monospace;font-size:.6rem;color:var(--accent);
  letter-spacing:.22em;text-transform:uppercase;border-bottom:1px solid var(--border);
  padding-bottom:.45rem;margin-bottom:.9rem;}

/* OUTLOOK BOX */
.outlook-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem;}
.ol-title{font-family:'Space Mono',monospace;font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;margin-bottom:.6rem;}
.ol-item{font-size:.78rem;color:var(--text);padding:3px 0;border-bottom:1px solid var(--border);}
.ol-item:last-child{border-bottom:none;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--surface);border-radius:8px;
  padding:3px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'Space Mono',monospace;font-size:.65rem;
  letter-spacing:.08em;color:var(--muted)!important;}
.stTabs [aria-selected="true"]{background:var(--card)!important;
  color:var(--accent)!important;border-radius:5px;}

/* SCROLLBAR */
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL MODEL — oil_return + gold_return → Risk ON/OFF
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """
    Preserves original logic from train_model.py:
      features = ["oil_return", "gold_return"]
      target   = sp500_return > 0  →  1 = Risk ON, 0 = Risk OFF
    When model.pkl is absent (e.g. Streamlit Cloud), trains on
    synthetic data that mirrors the same market relationship.
    """
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    np.random.seed(42)
    n = 2000
    # Risk ON: oil up, gold flat/down → sp500 likely positive
    # Risk OFF: oil down, gold up → sp500 likely negative
    oil_return  = np.concatenate([
        np.random.normal( 0.015, 0.025, n // 2),   # Risk ON
        np.random.normal(-0.015, 0.025, n // 2),   # Risk OFF
    ])
    gold_return = np.concatenate([
        np.random.normal(-0.005, 0.015, n // 2),   # Risk ON
        np.random.normal( 0.012, 0.015, n // 2),   # Risk OFF
    ])
    target = np.array([1] * (n // 2) + [0] * (n // 2))

    df = pd.DataFrame({
        "oil_return":  oil_return,
        "gold_return": gold_return,
        "target":      target,
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    # Same architecture as original train_model.py
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(df[["oil_return", "gold_return"]], df["target"])
    joblib.dump(model, "model.pkl")
    return model

model = load_or_train_model()

# ─────────────────────────────────────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
STOCKS = {
    "SPY":  "S&P 500 ETF",
    "QQQ":  "Nasdaq 100 ETF",
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corp.",
    "MSFT": "Microsoft",
    "TSLA": "Tesla Inc.",
    "GLD":  "Gold ETF",
    "USO":  "Oil ETF",
    "JPM":  "JPMorgan Chase",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "DIA":  "Dow Jones ETF",
}
CRYPTO = {
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "BNB-USD":  "BNB",
    "SOL-USD":  "Solana",
    "XRP-USD":  "XRP",
    "ADA-USD":  "Cardano",
    "AVAX-USD": "Avalanche",
    "DOGE-USD": "Dogecoin",
    "DOT-USD":  "Polkadot",
    "LINK-USD": "Chainlink",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
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
                "price":    price,
                "chg_pct":  chg_pct,
                "wk_chg":   wk_chg,
                "mo_chg":   mo_chg,
                "high":     float(hist["High"].iloc[-1]),
                "low":      float(hist["Low"].iloc[-1]),
                "volume":   float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
                "history":  hist["Close"].values.tolist(),
                "dates":    [str(d.date()) for d in hist.index],
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
    """
    Fetch real news headlines from yfinance for a given ticker.
    Returns list of dicts: {title, publisher, link, sentiment, signal}
    """
    items = []
    try:
        t   = yf.Ticker(ticker)
        raw = t.news or []
        BULL_WORDS = ["surge","jump","rally","gain","rise","beat","record","strong",
                      "buy","upgrade","bullish","profit","growth","soar","breakout","high"]
        BEAR_WORDS = ["drop","fall","crash","plunge","loss","miss","weak","sell",
                      "downgrade","bearish","decline","slump","risk","low","cut","warning"]
        for n in raw[:6]:
            title = n.get("title","")
            pub   = n.get("publisher","")
            link  = n.get("link","#")
            t_low = title.lower()
            bull_hits = sum(1 for w in BULL_WORDS if w in t_low)
            bear_hits = sum(1 for w in BEAR_WORDS if w in t_low)
            if bull_hits > bear_hits:
                sentiment = "bullish"
                signal    = "BUY"
            elif bear_hits > bull_hits:
                sentiment = "bearish"
                signal    = "SELL"
            else:
                sentiment = "neutral"
                signal    = "WATCH"
            items.append({
                "title":     title,
                "publisher": pub,
                "link":      link,
                "sentiment": sentiment,
                "signal":    signal,
            })
    except Exception:
        pass
    return items

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL SIGNAL ENGINE
# Uses price history + moving averages + momentum
# ─────────────────────────────────────────────────────────────────────────────
def compute_technical_signal(history, chg_pct, wk_chg):
    if len(history) < 5:
        return "HOLD", 50, "Insufficient data"
    h    = np.array(history)
    ma5  = np.mean(h[-5:])
    ma20 = np.mean(h[-20:]) if len(h) >= 20 else np.mean(h)
    ma50 = np.mean(h[-50:]) if len(h) >= 50 else np.mean(h)
    price = h[-1]
    score = 50

    # Trend
    if price > ma5:   score += 8
    if price > ma20:  score += 8
    if price > ma50:  score += 6
    if ma5  > ma20:   score += 8
    if ma20 > ma50:   score += 5

    # Momentum
    score += min(max(chg_pct * 2.5, -12), 12)
    score += min(max(wk_chg  * 1.2, -8),  8)

    # RSI-lite (last 14 days)
    if len(h) >= 15:
        deltas = np.diff(h[-15:])
        gains  = deltas[deltas > 0].mean() if any(deltas > 0) else 0
        losses = abs(deltas[deltas < 0].mean()) if any(deltas < 0) else 0.001
        rs     = gains / losses
        rsi    = 100 - (100 / (1 + rs))
        if rsi < 30:   score += 10   # oversold → buy signal
        elif rsi > 70: score -= 10   # overbought → sell signal

    score = max(0, min(100, int(score)))

    if score >= 63:
        reasoning = f"Price above MA5/MA20. Positive momentum. Score {score}/100."
        return "BUY", score, reasoning
    elif score <= 40:
        reasoning = f"Price below moving averages. Negative momentum. Score {score}/100."
        return "SELL", score, reasoning
    else:
        reasoning = f"Mixed signals. Consolidating. Score {score}/100."
        return "HOLD", score, reasoning

def news_signal(news_items):
    """Derive a news-based signal from headline sentiment."""
    if not news_items:
        return "NEUTRAL", 0, 0
    bull = sum(1 for n in news_items if n["sentiment"] == "bullish")
    bear = sum(1 for n in news_items if n["sentiment"] == "bearish")
    total = len(news_items)
    if bull > bear:
        return "BULLISH", bull, total
    elif bear > bull:
        return "BEARISH", bear, total
    else:
        return "NEUTRAL", 0, total

def combined_signal(tech_sig, tech_score, news_sent):
    """
    Combines technical signal with news sentiment.
    News acts as a confirming or contradicting factor.
    """
    score = tech_score
    if news_sent == "BULLISH":  score += 8
    elif news_sent == "BEARISH": score -= 8
    score = max(0, min(100, score))
    if score >= 60:   return "BUY",  score
    elif score <= 42: return "SELL", score
    else:             return "HOLD", score

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PBASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#4a6080", size=10),
    margin=dict(l=8, r=8, t=30, b=8),
    hovermode="x unified",
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
        fill="tozeroy",
        fillcolor=hex_to_rgba(color),
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=48,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

def candlestick_chart(ticker, title):
    df = fetch_ohlc(ticker)
    if df.empty:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00e676",  decreasing_fillcolor="#ff1744",
        name="OHLC",
    ), row=1, col=1)
    bar_colors = ["#00e676" if c >= o else "#ff1744"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=bar_colors, opacity=0.5, name="Vol",
    ), row=2, col=1)
    # MA overlays on candlestick
    close = df["Close"]
    if len(close) >= 10:
        fig.add_trace(go.Scatter(
            x=df.index, y=close.rolling(10).mean(),
            line=dict(color="#00c6ff", width=1, dash="dot"),
            name="MA10", hoverinfo="skip",
        ), row=1, col=1)
    if len(close) >= 20:
        fig.add_trace(go.Scatter(
            x=df.index, y=close.rolling(20).mean(),
            line=dict(color="#ffd600", width=1, dash="dot"),
            name="MA20", hoverinfo="skip",
        ), row=1, col=1)
    fig.update_layout(
        **PBASE, height=340,
        title=dict(text=title, font=dict(size=11, color="#00c6ff")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def normalised_chart(data_dict, title):
    COLORS = ["#00c6ff","#00e676","#ff1744","#ffd600","#a855f7",
              "#ff6d00","#06b6d4","#84cc16","#f43f5e","#8b5cf6",
              "#fb923c","#34d399"]
    fig = go.Figure()
    for i, (sym, d) in enumerate(data_dict.items()):
        h = np.array(d["history"])
        if len(h) < 2 or h[0] == 0:
            continue
        norm = (h / h[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=d["dates"], y=np.round(norm, 2), name=sym, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate=f"<b>{sym}</b>: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        **PBASE, height=360,
        title=dict(text=title, font=dict(size=11, color="#00c6ff")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading live market data…"):
    all_tickers = list(STOCKS.keys()) + list(CRYPTO.keys())
    prices      = fetch_market_data(all_tickers)

now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <p class="hdr-title">📡 Geo Market Intelligence</p>
  <p class="hdr-sub">
    <span class="live"></span>Live &nbsp;·&nbsp; {now_str}
    &nbsp;·&nbsp; 12 Stocks &nbsp;·&nbsp; 10 Crypto &nbsp;·&nbsp;
    Oil+Gold Risk Engine &nbsp;·&nbsp; News Signals
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL RISK ENGINE  (oil_return + gold_return → model.predict)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">⚡ Risk Engine — Original Oil + Gold Model</p>', unsafe_allow_html=True)

uso_d  = prices.get("USO", {})
gld_d  = prices.get("GLD", {})
spy_d  = prices.get("SPY", {})

oil_return  = uso_d.get("chg_pct",  0.0) / 100   # same feature name as original
gold_return = gld_d.get("chg_pct",  0.0) / 100   # same feature name as original

# Exact same predict call as original predict.py
today_input = pd.DataFrame([{
    "oil_return":  oil_return,
    "gold_return": gold_return,
}])
prediction   = model.predict(today_input)
proba        = model.predict_proba(today_input)[0]
conf         = int(max(proba) * 100)

risk_on      = prediction[0] == 1
risk_label   = "RISK ON — Markets Bullish"  if risk_on else "RISK OFF — Markets Defensive"
risk_emoji   = "🟢" if risk_on else "🔴"
risk_banner  = "risk-on-banner" if risk_on else "risk-off-banner"
risk_color   = "#00e676" if risk_on else "#ff1744"

if risk_on:
    risk_desc = ("Oil is rising and gold is flat/falling — investors are risk-seeking. "
                 "Conditions favour equities, growth assets, and cyclicals. "
                 "Watch for follow-through in SPY, QQQ, and high-beta names.")
    horizon_24h = "📈 Upside bias expected in next 24 hours. Risk assets likely to extend gains."
    horizon_7d  = "📈 Bullish structure intact if oil holds. Watch for pullback entry on dips."
else:
    risk_desc = ("Oil is falling and gold is rising — investors are moving to safe havens. "
                 "Conditions favour defensive positioning: gold, bonds, cash, or short exposure. "
                 "Equities and crypto face headwinds.")
    horizon_24h = "📉 Downside bias expected in next 24 hours. Risk assets may continue declining."
    horizon_7d  = "📉 Defensive structure. Avoid chasing rallies — wait for risk appetite to recover."

r1, r2, r3, r4, r5 = st.columns([2, 1, 1, 1, 1])
with r1:
    st.markdown(f"""
    <div class="{risk_banner}">
      <div class="risk-title" style="color:{risk_color}">{risk_emoji} {risk_label}</div>
      <div class="risk-conf">Confidence: {conf}%</div>
      <div class="risk-desc">{risk_desc}</div>
    </div>""", unsafe_allow_html=True)

for col, label, val, status, color in [
    (r2, "OIL RETURN",  f"{oil_return*100:+.2f}%",
     "▲ Bullish" if oil_return > 0 else "▼ Bearish",
     "#00e676" if oil_return > 0 else "#ff1744"),
    (r3, "GOLD RETURN", f"{gold_return*100:+.2f}%",
     "▲ Safe Haven" if gold_return > 0.003 else "▼ Risk Seek",
     "#ff1744" if gold_return > 0.003 else "#00e676"),
    (r4, "MODEL INPUT",  "oil + gold", "RandomForest 200", "#00c6ff"),
    (r5, "24H OUTLOOK",  "→ See below", horizon_24h[:28]+"…", "#ffd600"),
]:
    with col:
        st.markdown(f"""
        <div class="ind-tile">
          <div class="ind-label">{label}</div>
          <div class="ind-val">{val}</div>
          <div class="ind-status" style="color:{color}">{status}</div>
        </div>""", unsafe_allow_html=True)

# 24h / 7d outlook strip
st.markdown(f"""
<div style="display:flex;gap:.8rem;margin:.8rem 0 1.2rem">
  <div style="flex:1;background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:.7rem 1rem;font-size:.78rem;line-height:1.5">
    <span style="font-family:'Space Mono';font-size:.58rem;color:var(--accent);
      letter-spacing:.15em">24-HOUR OUTLOOK</span><br>{horizon_24h}
  </div>
  <div style="flex:1;background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:.7rem 1rem;font-size:.78rem;line-height:1.5">
    <span style="font-family:'Space Mono';font-size:.58rem;color:var(--accent);
      letter-spacing:.15em">7-DAY OUTLOOK</span><br>{horizon_7d}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  STOCKS",
    "₿   CRYPTO",
    "📰  NEWS SIGNALS",
    "📊  COMPARE",
])

# ── Helper: render asset grid ─────────────────────────────────────────────────
def render_asset_grid(asset_dict, prices):
    cols = st.columns(4)
    for i, (ticker, name) in enumerate(asset_dict.items()):
        d = prices.get(ticker)
        if not d:
            continue
        tech_sig, tech_score, tech_reason = compute_technical_signal(
            d["history"], d["chg_pct"], d["wk_chg"]
        )
        news_items  = fetch_news(ticker)
        news_sent, news_count, news_total = news_signal(news_items)
        final_sig, final_score = combined_signal(tech_sig, tech_score, news_sent)

        chg   = d["chg_pct"]
        sym   = ticker.replace("-USD", "")
        p     = d["price"]
        pfmt  = (f"${p:,.0f}" if p > 500 else
                 f"${p:,.2f}" if p > 1   else f"${p:.5f}")
        chg_cls  = "ac-chg-pos" if chg >= 0 else "ac-chg-neg"
        card_cls = "ac-bull" if final_sig=="BUY" else ("ac-bear" if final_sig=="SELL" else "ac-hold")
        sig_cls  = f"sig-{final_sig.lower()}"

        with cols[i % 4]:
            st.markdown(f"""
            <div class="asset-card {card_cls}">
              <div class="ac-top">
                <div>
                  <div class="ac-sym">{sym}</div>
                  <div class="ac-name">{name}</div>
                </div>
                <span class="{sig_cls}">{final_sig}</span>
              </div>
              <div style="display:flex;align-items:baseline;gap:7px;margin-top:.5rem">
                <span class="ac-price">{pfmt}</span>
                <span class="{chg_cls}">{"▲" if chg>=0 else "▼"}{abs(chg):.2f}%</span>
              </div>
              <div class="ac-meta">
                7D:{d['wk_chg']:+.1f}%
                &nbsp;·&nbsp;30D:{d['mo_chg']:+.1f}%
                &nbsp;·&nbsp;Score:{final_score}/100
              </div>
              <div class="ac-meta" style="margin-top:.3rem;color:#4a6080;font-size:.55rem">
                Tech:{tech_sig} · News:{news_sent} · {news_total} headlines
              </div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(
                sparkline(d["history"], "#00e676" if chg >= 0 else "#ff1744"),
                use_container_width=True, config={"displayModeBar": False},
            )

# ── Helper: buy/sell/hold summary ─────────────────────────────────────────────
def render_outlook_summary(asset_dict, prices, label):
    buys, sells, holds = [], [], []
    for ticker in asset_dict:
        d = prices.get(ticker)
        if not d:
            continue
        tech_sig, tech_score, _ = compute_technical_signal(d["history"], d["chg_pct"], d["wk_chg"])
        news_items = fetch_news(ticker)
        ns, _, _ = news_signal(news_items)
        sig, score = combined_signal(tech_sig, tech_score, ns)
        sym = ticker.replace("-USD","")
        entry = f"{sym} ({score}/100)"
        if sig == "BUY":   buys.append(entry)
        elif sig == "SELL": sells.append(entry)
        else:               holds.append(entry)

    c1, c2, c3 = st.columns(3)
    none_txt = '<div style="color:#4a6080;font-size:.73rem">— None currently —</div>'
    for col, title, color, items, cls in [
        (c1, "✅  BUY — Enter Long",      "#00e676", buys,  "ac-bull"),
        (c2, "🔴  SELL — Exit / Short",   "#ff1744", sells, "ac-bear"),
        (c3, "⏸  HOLD — Watch & Wait",   "#ffd600", holds, "ac-hold"),
    ]:
        with col:
            rows = "".join([f'<div class="ol-item">→ {it}</div>' for it in items]) or none_txt
            st.markdown(f"""
            <div class="outlook-box {cls}">
              <div class="ol-title" style="color:{color}">{title}</div>
              {rows}
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — STOCKS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="sec">Stock Assets — Technical + News Signal</p>', unsafe_allow_html=True)
    render_asset_grid(STOCKS, prices)

    st.markdown('<p class="sec" style="margin-top:1.2rem">Candlestick — 5-Day Hourly with MA Overlays</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, tk, title in [
        (c1, "SPY",  "SPY — S&P 500"),
        (c2, "QQQ",  "QQQ — Nasdaq 100"),
    ]:
        with col:
            st.plotly_chart(candlestick_chart(tk, title),
                            use_container_width=True, config={"displayModeBar": False})
    c3, c4 = st.columns(2)
    for col, tk, title in [
        (c3, "AAPL", "AAPL — Apple"),
        (c4, "NVDA", "NVDA — NVIDIA"),
    ]:
        with col:
            st.plotly_chart(candlestick_chart(tk, title),
                            use_container_width=True, config={"displayModeBar": False})

    st.markdown('<p class="sec" style="margin-top:1.2rem">📋 Stock Outlook — Where Market Is Leading</p>', unsafe_allow_html=True)
    render_outlook_summary(STOCKS, prices, "Stocks")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CRYPTO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="sec">Digital Assets — Technical + News Signal</p>', unsafe_allow_html=True)
    render_asset_grid(CRYPTO, prices)

    st.markdown('<p class="sec" style="margin-top:1.2rem">Candlestick — BTC & ETH Hourly</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(candlestick_chart("BTC-USD", "BTC — Bitcoin"),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.plotly_chart(candlestick_chart("ETH-USD", "ETH — Ethereum"),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown('<p class="sec" style="margin-top:1.2rem">📋 Crypto Outlook — Where Market Is Leading</p>', unsafe_allow_html=True)
    render_outlook_summary(CRYPTO, prices, "Crypto")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — NEWS SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="sec">News-Based Signals — Real Headlines Per Asset</p>', unsafe_allow_html=True)

    pick_col1, pick_col2 = st.columns([1, 3])
    with pick_col1:
        category = st.radio("Asset class", ["Stocks", "Crypto"], horizontal=False)
    with pick_col2:
        pool = STOCKS if category == "Stocks" else CRYPTO
        selected = st.selectbox(
            "Select asset",
            list(pool.keys()),
            format_func=lambda t: f"{t.replace('-USD','')} — {pool[t]}"
        )

    if selected:
        d_sel = prices.get(selected, {})
        news  = fetch_news(selected)

        # Asset quick stats
        if d_sel:
            tech_sig, tech_score, tech_reason = compute_technical_signal(
                d_sel["history"], d_sel["chg_pct"], d_sel["wk_chg"]
            )
            ns, nc, nt = news_signal(news)
            fsig, fscore = combined_signal(tech_sig, tech_score, ns)
            p = d_sel["price"]
            pfmt = f"${p:,.2f}" if p < 10000 else f"${p:,.0f}"

            m1,m2,m3,m4,m5 = st.columns(5)
            for col, lbl, val, clr in [
                (m1, "PRICE",        pfmt,                               "#dde8f5"),
                (m2, "24H CHANGE",   f"{d_sel['chg_pct']:+.2f}%",       "#00e676" if d_sel['chg_pct']>=0 else "#ff1744"),
                (m3, "TECH SIGNAL",  tech_sig,                           "#00e676" if tech_sig=="BUY" else "#ff1744" if tech_sig=="SELL" else "#ffd600"),
                (m4, "NEWS MOOD",    ns,                                 "#00e676" if ns=="BULLISH" else "#ff1744" if ns=="BEARISH" else "#4a6080"),
                (m5, "FINAL SIGNAL", fsig,                               "#00e676" if fsig=="BUY" else "#ff1744" if fsig=="SELL" else "#ffd600"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="ind-tile">
                      <div class="ind-label">{lbl}</div>
                      <div class="ind-val" style="font-size:1.1rem;color:{clr}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;
              padding:.75rem 1rem;margin:.7rem 0;font-size:.78rem;color:#dde8f5;line-height:1.55">
              <span style="font-family:'Space Mono';font-size:.58rem;color:var(--accent);
                letter-spacing:.15em">TECHNICAL REASONING</span><br>{tech_reason}
            </div>""", unsafe_allow_html=True)

        # News headlines
        st.markdown('<p class="sec" style="margin-top:.8rem">Latest News Headlines</p>', unsafe_allow_html=True)
        if news:
            for n in news:
                sent   = n["sentiment"]
                ncls   = f"news-{sent[:4]}"     # news-bull / news-bear / news-neut
                scls   = f"news-sentiment-{sent[:4]}"
                emoji  = "📈" if sent=="bullish" else ("📉" if sent=="bearish" else "➡️")
                label  = sent.upper()
                st.markdown(f"""
                <div class="news-item {ncls}">
                  <div class="news-src">{n['publisher']}</div>
                  <div class="news-headline">{n['title']}</div>
                  <div class="{scls}">{emoji} {label} &nbsp;·&nbsp; Signal: <b>{n['signal']}</b></div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="news-item news-neut">
              <div class="news-headline" style="color:#4a6080">
                No recent headlines found for this asset via Yahoo Finance.
              </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COMPARE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="sec">60-Day Normalised Performance</p>', unsafe_allow_html=True)

    sd = {t: prices[t] for t in STOCKS if t in prices and len(prices[t]["history"]) > 1}
    cd = {t.replace("-USD",""): prices[t] for t in CRYPTO
          if t in prices and len(prices[t]["history"]) > 1}

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(normalised_chart(sd, "Stocks — 60D % Return"),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.plotly_chart(normalised_chart(cd, "Crypto — 60D % Return"),
                        use_container_width=True, config={"displayModeBar": False})

    # Full scorecard
    st.markdown('<p class="sec" style="margin-top:1rem">Full Asset Scorecard</p>', unsafe_allow_html=True)
    rows = []
    for ticker, name in {**STOCKS, **CRYPTO}.items():
        d = prices.get(ticker)
        if not d:
            continue
        ts, score, _ = compute_technical_signal(d["history"], d["chg_pct"], d["wk_chg"])
        ni = fetch_news(ticker)
        ns, _, _ = news_signal(ni)
        fsig, fscore = combined_signal(ts, score, ns)
        p = d["price"]
        rows.append({
            "Asset":       ticker.replace("-USD",""),
            "Name":        name,
            "Price":       f"${p:,.2f}" if p < 100000 else f"${p:,.0f}",
            "24H %":       f"{d['chg_pct']:+.2f}%",
            "7D %":        f"{d['wk_chg']:+.2f}%",
            "30D %":       f"{d['mo_chg']:+.2f}%",
            "Tech":        ts,
            "News Mood":   ns,
            "Signal":      fsig,
            "Score":       fscore,
        })

    def highlight_cell(val):
        if val in ("BUY","BULLISH"):   return "color:#00e676;font-weight:bold"
        if val in ("SELL","BEARISH"):  return "color:#ff1744;font-weight:bold"
        if val in ("HOLD","NEUTRAL"):  return "color:#ffd600;font-weight:bold"
        if isinstance(val, str) and val.startswith("+"): return "color:#00e676"
        if isinstance(val, str) and val.startswith("-"): return "color:#ff1744"
        return ""

    df_out = pd.DataFrame(rows)
    st.dataframe(
        df_out.style.applymap(
            highlight_cell,
            subset=["Signal","Tech","News Mood","24H %","7D %","30D %"]
        ),
        use_container_width=True, hide_index=True, height=680,
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2.5rem;padding:1rem 0;border-top:1px solid #1a2840;
  font-family:'Space Mono',monospace;font-size:.58rem;color:#2a4060;
  text-align:center;letter-spacing:.15em">
  GEO MARKET INTELLIGENCE &nbsp;·&nbsp; ORIGINAL LOGIC: OIL + GOLD → RANDOMFOREST
  &nbsp;·&nbsp; DATA: YAHOO FINANCE &nbsp;·&nbsp; REFRESHES EVERY 5 MIN
  &nbsp;·&nbsp; NOT FINANCIAL ADVICE &nbsp;·&nbsp; {now_str}
</div>
""", unsafe_allow_html=True)
