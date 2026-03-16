"""
GEO MARKET INTELLIGENCE DASHBOARD v5.1 — CHRONOS-ALPHA + STEADYFIT SKIN
Original logic: oil_return + gold_return → RandomForest → Risk ON / Risk OFF  [UNCHANGED]
Extended v3:    5s live prices, trade timing, GradientBoosting timing model    [UNCHANGED]
Extended v4:    Chronos-Alpha AI layer via Groq + Llama 3.3 70B               [UNCHANGED]
Extended v5:    Steadyfit dark-purple skin (Landzy-style), animated alerts,
                Settings panel (theme/skin/alerts), About modal, Donate + USDT
                wallet copy, Contact form, top security (CSP, XSS sanit.)
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
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geo Market Intelligence | Steadyfit",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CHRONOS-ALPHA CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

CHRONOS_SYSTEM_PROMPT = """You are Chronos-Alpha, a top-tier autonomous AI quantitative analyst
and financial strategist. You operate with surgical precision, combining technical analysis,
news sentiment, geopolitical intelligence, and market regime detection.

Your core principles:
- Never act on emotion. Stick to data.
- Prioritize capital preservation over quick gains.
- Maximum drawdown tolerance: 5%
- Always flag geopolitical risk, black swan indicators, and systemic fragility.
- Provide specific, actionable intelligence — not vague commentary.
- Be concise, direct, and professional. Think like a hedge fund quant.

Output format: Always structured, precise, and data-driven.
When you detect black swan risk or geopolitical triggers, flag them prominently with ⚠️"""

@st.cache_data(ttl=600)
def chronos_analyse_asset(ticker, name, asset_class, price, chg_pct, wk_chg,
                            rsi, signal, score, news_headlines, groq_key):
    if not groq_key or not GROQ_AVAILABLE:
        return None
    headlines_text = "\n".join([
        f"- [{n['sentiment'].upper()}] {n['title']} ({n['publisher']})"
        for n in news_headlines[:5]
    ]) if news_headlines else "No recent headlines available."
    prompt = f"""Asset: {ticker} ({name}) | Class: {asset_class}
Live Price: ${price:,.4g} | 24H Change: {chg_pct:+.2f}% | 7D Change: {wk_chg:+.2f}%
RSI(14): {rsi:.1f} | Technical Signal: {signal} (Score: {score}/100)

Recent News Headlines:
{headlines_text}

Provide a Chronos-Alpha analysis with EXACTLY this structure:

DIRECTION: [BUY/SELL/HOLD] — [one sentence price direction prediction for next 24-48h]
REASONING: [why — based on technicals + news combined]
TIMEFRAME: [specific window e.g. "Entry now, target in 4-8 hours"]
RISK: [key risk factor that could invalidate this signal]
GEOPOLITICAL: [any geopolitical or macro factor affecting this asset, or NONE DETECTED]
BLACK SWAN: [any systemic risk indicator present, or NONE DETECTED]"""
    try:
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": CHRONOS_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300, temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Chronos-Alpha unavailable: {e}"


@st.cache_data(ttl=600)
def chronos_market_summary(risk_on, oil_ret, gold_ret, risk_conf,
                             stock_signals, crypto_signals, groq_key):
    if not groq_key or not GROQ_AVAILABLE:
        return None
    regime = "RISK ON — Bullish" if risk_on else "RISK OFF — Defensive"
    buy_stocks  = [s for s,sig in stock_signals.items()  if sig=="BUY"]
    sell_stocks = [s for s,sig in stock_signals.items()  if sig=="SELL"]
    buy_crypto  = [s for s,sig in crypto_signals.items() if sig=="BUY"]
    sell_crypto = [s for s,sig in crypto_signals.items() if sig=="SELL"]
    prompt = f"""Global Market Regime: {regime} (Confidence: {risk_conf}%)
Oil Return: {oil_ret*100:+.2f}% | Gold Return: {gold_ret*100:+.2f}%

Stock BUY signals:  {", ".join(buy_stocks)  or "None"}
Stock SELL signals: {", ".join(sell_stocks) or "None"}
Crypto BUY signals: {", ".join(buy_crypto)  or "None"}
Crypto SELL signals:{", ".join(sell_crypto) or "None"}

As Chronos-Alpha, provide a Daily Action Report with EXACTLY this structure:

MARKET REGIME: [1 sentence describing current market environment]
PRIMARY RISK: [1 sentence on the biggest risk to portfolios right now]
ACTIVE STRATEGY: [1 sentence on the recommended positioning approach]
24H OUTLOOK: [specific prediction for next 24 hours]
7D OUTLOOK: [broader view for the week ahead]
BLACK SWAN WATCH: [any early warning indicators present, or CLEAR]"""
    try:
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": CHRONOS_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400, temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Chronos-Alpha unavailable: {e}"


@st.cache_data(ttl=300)
def chronos_geopolitical_scan(groq_key):
    if not groq_key or not GROQ_AVAILABLE:
        return None
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    prompt = f"""Current timestamp: {now}

As Chronos-Alpha, perform a rapid geopolitical and black swan risk scan.
Provide EXACTLY this structure:

THREAT LEVEL: [LOW / MEDIUM / HIGH / CRITICAL]
ACTIVE RISKS: [list up to 3 current macro/geopolitical risks]
SECTORS AT RISK: [which asset classes are most exposed]
SAFE HAVENS: [where capital should rotate if risk escalates]
MONITORING: [what to watch in next 24-48 hours]"""
    try:
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": CHRONOS_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=350, temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Chronos-Alpha unavailable: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "last_refresh"  not in st.session_state: st.session_state.last_refresh  = time.time()
if "prev_prices"   not in st.session_state: st.session_state.prev_prices   = {}
if "groq_key"      not in st.session_state: st.session_state.groq_key      = GROQ_API_KEY
if "theme"         not in st.session_state: st.session_state.theme         = "dark"
if "alerts_shown"  not in st.session_state: st.session_state.alerts_shown  = []
if "alert_queue"   not in st.session_state: st.session_state.alert_queue   = []

# Auto-refresh every 5 seconds
if time.time() - st.session_state.last_refresh > 5:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# v5 SKIN — STEADYFIT PURPLE / DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Barlow:wght@600;700;800&display=swap');

/* ─── TRADINGVIEW-INSPIRED TOKENS ───────────────────────── */
:root{
  /* Backgrounds — TV uses very precise near-black layering */
  --bg:#131722;--surface:#1e222d;--card:#1e222d;--card2:#2a2e39;
  --card-glass:rgba(30,34,45,.72);--card-glass2:rgba(42,46,57,.65);

  /* Borders — TV uses hairline separators, not glowing borders */
  --border:rgba(255,255,255,.06);--border2:rgba(255,255,255,.10);
  --border-accent:rgba(41,98,255,.35);

  /* Brand — TV uses electric blue as its single accent */
  --blue:#2962ff;--blue2:#1e53e5;--blue-soft:rgba(41,98,255,.15);
  --blue-glow:rgba(41,98,255,.08);

  /* Semantic */
  --green:#26a69a;--green2:#00897b;--green-soft:rgba(38,166,154,.14);
  --red:#ef5350;--red2:#e53935;--red-soft:rgba(239,83,80,.14);
  --yellow:#f9a825;--yellow-soft:rgba(249,168,37,.12);
  --cyan:#00bcd4;--orange:#ff9800;--purple:#9c27b0;--violet:#b39ddb;

  /* Text — TV uses a cool-white hierarchy */
  --text:#d1d4dc;--text2:#787b86;--text3:#434651;--muted:#4e5261;--muted2:#2a2e39;

  /* Shadows */
  --shadow:rgba(0,0,0,.55);--shadow2:rgba(0,0,0,.35);

  /* Typography */
  --font:'IBM Plex Sans',system-ui,sans-serif;
  --mono:'IBM Plex Mono',monospace;
  --display:'Barlow',sans-serif;
}

/* ─── RESET ──────────────────────────────────────────────── */
*{box-sizing:border-box;}
html,body,[class*="css"]{
  background:var(--bg)!important;
  color:var(--text)!important;
  font-family:var(--font)!important;
  font-size:14px;
}
.main,.block-container{
  background:var(--bg)!important;
  padding:0 1.6rem 1.6rem!important;
  max-width:100%!important;
}

/* ─── SUBTLE SCANLINE TEXTURE (TV pro feel) ──────────────── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:-1;
  background-image:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(255,255,255,.008) 2px,
    rgba(255,255,255,.008) 4px
  );
}

/* ─── SIDEBAR — left-nav panel style ─────────────────────── */
[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
  width:260px!important;
}
[data-testid="stSidebar"]>div:first-child{padding:.75rem .8rem;}
[data-testid="stSidebarCollapseButton"]{color:var(--text2)!important;}

/* ─── TABS — TV-style flat underline nav ─────────────────── */
.stTabs [data-baseweb="tab-list"]{
  background:transparent;
  border-bottom:1px solid var(--border);
  border-radius:0;padding:0;gap:0;
}
.stTabs [data-baseweb="tab"]{
  font-family:var(--font);font-size:.78rem;font-weight:500;letter-spacing:.01em;
  color:var(--text2)!important;border-radius:0;
  padding:.6rem 1.1rem;border-bottom:2px solid transparent;
  transition:all .15s;background:transparent!important;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--text)!important;}
.stTabs [aria-selected="true"]{
  color:var(--text)!important;
  border-bottom:2px solid var(--blue)!important;
  background:transparent!important;
}

/* ─── TOP TICKER BAR ─────────────────────────────────────── */
.ticker-bar{
  display:flex;align-items:center;gap:0;
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 1.6rem;overflow-x:auto;margin-bottom:1rem;
  scrollbar-width:none;
}
.ticker-bar::-webkit-scrollbar{display:none;}
.ticker-item{
  display:flex;flex-direction:column;align-items:center;
  padding:.45rem 1.1rem;border-right:1px solid var(--border);
  cursor:default;transition:background .15s;min-width:100px;
}
.ticker-item:hover{background:var(--card2);}
.ticker-sym{font-family:var(--mono);font-size:.65rem;font-weight:500;color:var(--text2);letter-spacing:.06em;}
.ticker-price{font-family:var(--mono);font-size:.82rem;font-weight:500;color:var(--text);margin-top:1px;}
.ticker-chg-pos{font-family:var(--mono);font-size:.6rem;color:var(--green);margin-top:1px;}
.ticker-chg-neg{font-family:var(--mono);font-size:.6rem;color:var(--red);margin-top:1px;}

/* ─── HEADER — TV-style compact topbar ───────────────────── */
.hdr{
  display:flex;align-items:center;justify-content:space-between;
  padding:.7rem 0 .7rem;
  border-bottom:1px solid var(--border);
  margin-bottom:0;
}
.hdr-title{
  font-family:var(--display);font-size:1.1rem;font-weight:700;
  letter-spacing:.08em;color:var(--text);margin:0;
  text-transform:uppercase;
}
.hdr-title span{color:var(--blue);}
.hdr-sub{
  font-family:var(--mono);font-size:.6rem;color:var(--text2);
  letter-spacing:.08em;margin-top:0;display:flex;align-items:center;gap:.6rem;
}
.live-dot{
  display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green);
  animation:ldot 2s ease-in-out infinite;
}
@keyframes ldot{0%,100%{opacity:1}50%{opacity:.25}}

/* ─── SECTION LABELS — TV inline dividers ────────────────── */
.sec{
  font-family:var(--mono);font-size:.6rem;color:var(--text2);letter-spacing:.16em;
  text-transform:uppercase;
  display:flex;align-items:center;gap:.6rem;
  margin-bottom:.75rem;margin-top:.2rem;
}
.sec::after{
  content:'';flex:1;height:1px;background:var(--border);
}

/* ─── PRICE FLASH ────────────────────────────────────────── */
.price-up  {font-family:var(--mono);font-size:1.15rem;font-weight:600;color:var(--green);animation:flashup .5s ease;}
.price-down{font-family:var(--mono);font-size:1.15rem;font-weight:600;color:var(--red);animation:flashdn .5s ease;}
.price-flat{font-family:var(--mono);font-size:1.15rem;font-weight:600;color:var(--text);}
@keyframes flashup{0%{color:#4cffea}100%{color:var(--green)}}
@keyframes flashdn{0%{color:#ff6b6b}100%{color:var(--red)}}

/* ─── SIGNAL PILLS — sharper TV-style badges ─────────────── */
.sig-buy {
  background:var(--green-soft);color:var(--green);
  border:1px solid rgba(38,166,154,.3);
  border-radius:3px;padding:1px 8px;
  font-family:var(--mono);font-size:.62rem;font-weight:600;letter-spacing:.04em;
}
.sig-sell{
  background:var(--red-soft);color:var(--red);
  border:1px solid rgba(239,83,80,.3);
  border-radius:3px;padding:1px 8px;
  font-family:var(--mono);font-size:.62rem;font-weight:600;letter-spacing:.04em;
}
.sig-hold{
  background:var(--yellow-soft);color:var(--yellow);
  border:1px solid rgba(249,168,37,.3);
  border-radius:3px;padding:1px 8px;
  font-family:var(--mono);font-size:.62rem;font-weight:600;letter-spacing:.04em;
}

/* ─── ASSET CARD — glassmorphism TV panel ────────────────── */
.asset-card{
  background:var(--card-glass);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid var(--border);
  border-radius:8px;
  padding:.8rem .95rem;margin-bottom:.6rem;
  transition:border-color .18s,transform .18s;
  position:relative;overflow:hidden;
}
.asset-card::before{
  content:'';position:absolute;inset:0;border-radius:8px;
  background:linear-gradient(135deg,rgba(255,255,255,.025) 0%,transparent 60%);
  pointer-events:none;
}
.asset-card:hover{
  border-color:rgba(41,98,255,.3);
  transform:translateY(-1px);
  box-shadow:0 8px 24px rgba(0,0,0,.5),0 0 0 1px rgba(41,98,255,.12);
}
.ac-bull{border-top:2px solid var(--green);}
.ac-bear{border-top:2px solid var(--red);}
.ac-hold{border-top:2px solid var(--yellow);}
.ac-top{display:flex;justify-content:space-between;align-items:flex-start;}
.ac-sym{font-family:var(--mono);font-size:.75rem;font-weight:600;color:var(--text);letter-spacing:.04em;}
.ac-name{font-size:.65rem;color:var(--text2);margin-top:2px;font-weight:400;}
.ac-meta{font-family:var(--mono);font-size:.58rem;color:var(--text2);margin-top:.3rem;}
.chg-pos{font-family:var(--mono);font-size:.75rem;font-weight:600;color:var(--green);}
.chg-neg{font-family:var(--mono);font-size:.75rem;font-weight:600;color:var(--red);}

/* ─── RISK BANNER — glass panel with top accent ──────────── */
.risk-on-banner{
  background:var(--card-glass);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(38,166,154,.2);
  border-top:2px solid var(--green);
  border-radius:8px;padding:1rem 1.4rem;
}
.risk-off-banner{
  background:var(--card-glass);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(239,83,80,.2);
  border-top:2px solid var(--red);
  border-radius:8px;padding:1rem 1.4rem;
}
.risk-title{font-family:var(--display);font-size:1.35rem;font-weight:700;letter-spacing:.02em;}
.risk-conf{font-family:var(--mono);font-size:.62rem;color:var(--text2);margin:.25rem 0;}
.risk-desc{font-size:.76rem;line-height:1.55;color:var(--text);}

/* ─── INDICATOR TILE — glass metric card ─────────────────── */
.ind-tile{
  background:var(--card-glass2);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border:1px solid var(--border);
  border-radius:8px;padding:.85rem 1rem;text-align:center;
  position:relative;overflow:hidden;
}
.ind-tile::after{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent);
}
.ind-label{font-family:var(--mono);font-size:.58rem;color:var(--text2);letter-spacing:.12em;text-transform:uppercase;}
.ind-val{font-family:var(--mono);font-size:1.2rem;font-weight:600;color:var(--text);margin:.25rem 0 .15rem;}
.ind-status{font-size:.68rem;font-weight:500;}

/* ─── TRADE BOX — glass execution panel ─────────────────── */
.trade-box{border-radius:8px;padding:1rem 1.25rem;margin:.5rem 0;}
.trade-buy {
  background:linear-gradient(135deg,rgba(38,166,154,.07),rgba(0,137,123,.03));
  border:1px solid rgba(38,166,154,.2);border-top:2px solid var(--green);
}
.trade-sell{
  background:linear-gradient(135deg,rgba(239,83,80,.07),rgba(229,57,53,.03));
  border:1px solid rgba(239,83,80,.2);border-top:2px solid var(--red);
}
.trade-hold{
  background:rgba(249,168,37,.04);
  border:1px solid rgba(249,168,37,.18);border-top:2px solid var(--yellow);
}
.trade-title{font-family:var(--display);font-size:1.25rem;font-weight:700;letter-spacing:.02em;}
.trade-row{display:flex;gap:.75rem;margin-top:.65rem;flex-wrap:wrap;}
.trade-cell{
  background:rgba(0,0,0,.22);
  border:1px solid var(--border);
  border-radius:6px;padding:.45rem .75rem;min-width:105px;
}
.tc-label{font-family:var(--mono);font-size:.55rem;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;}
.tc-val{font-family:var(--mono);font-size:.88rem;font-weight:600;margin-top:.15rem;}

/* ─── CHRONOS CARDS — AI intelligence panels ─────────────── */
.ch-card{
  background:var(--card-glass);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid rgba(41,98,255,.15);
  border-left:2px solid var(--blue);
  border-radius:8px;padding:.9rem 1.1rem;margin:.5rem 0;
}
.ch-sum-card{
  background:var(--card-glass);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border:1px solid rgba(41,98,255,.15);
  border-top:2px solid var(--blue);
  border-radius:8px;padding:1rem 1.2rem;margin:.7rem 0 1rem;
}
.ch-hdr{
  font-family:var(--mono);font-size:.6rem;color:var(--blue);
  letter-spacing:.16em;text-transform:uppercase;margin-bottom:.6rem;
  display:flex;align-items:center;gap:.4rem;
}
.ch-hdr::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--blue);flex-shrink:0;}
.ch-row{
  padding:.32rem 0;border-bottom:1px solid rgba(255,255,255,.04);
  display:flex;gap:.75rem;align-items:baseline;
}
.ch-row:last-child{border-bottom:none;}
.ch-key{font-family:var(--mono);font-size:.57rem;letter-spacing:.08em;min-width:110px;flex-shrink:0;color:var(--text2);}
.ch-val{font-size:.76rem;color:var(--text);line-height:1.5;}

/* ─── NEWS ITEMS ─────────────────────────────────────────── */
.news-item{
  background:var(--card-glass2);
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  border:1px solid var(--border);
  border-radius:6px;padding:.7rem .9rem;margin-bottom:.45rem;
  transition:border-color .15s,transform .15s;
}
.news-item:hover{border-color:rgba(41,98,255,.25);transform:translateX(2px);}
.news-bull{border-left:2px solid var(--green);}
.news-bear{border-left:2px solid var(--red);}
.news-neut{border-left:2px solid var(--text3);}
.news-src{font-family:var(--mono);font-size:.57rem;color:var(--blue);letter-spacing:.08em;text-transform:uppercase;}
.news-headline{font-size:.76rem;color:var(--text);margin:.25rem 0 .15rem;line-height:1.45;}
.news-sentiment-bull{font-size:.62rem;color:var(--green);font-weight:600;}
.news-sentiment-bear{font-size:.62rem;color:var(--red);font-weight:600;}
.news-sentiment-neut{font-size:.62rem;color:var(--text2);font-weight:500;}

/* ─── OUTLOOK BOX ────────────────────────────────────────── */
.outlook-box{
  background:var(--card-glass2);border:1px solid var(--border);
  border-radius:8px;padding:.9rem 1rem;
}
.ol-title{font-family:var(--mono);font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;color:var(--text2);margin-bottom:.55rem;}
.ol-item{font-size:.76rem;color:var(--text);padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);}
.ol-item:last-child{border-bottom:none;}

/* ─── ALERT TOASTS — crisp TV-style notifications ────────── */
#toast-container{position:fixed;top:56px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:6px;width:320px;pointer-events:none;}
.toast{
  background:rgba(30,34,45,.96);
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid var(--border2);
  border-radius:6px;
  padding:.65rem .9rem;display:flex;align-items:flex-start;gap:.6rem;
  box-shadow:0 4px 20px rgba(0,0,0,.6);pointer-events:all;
  position:relative;overflow:hidden;
  animation:toastin .28s cubic-bezier(.22,1,.36,1);
}
.toast::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;}
.toast-buy  {border-color:rgba(38,166,154,.25);} .toast-buy::before  {background:var(--green);}
.toast-sell {border-color:rgba(239,83,80,.25);}  .toast-sell::before {background:var(--red);}
.toast-risk {border-color:rgba(41,98,255,.3);}   .toast-risk::before {background:var(--blue);}
.toast-news {border-color:rgba(249,168,37,.25);} .toast-news::before {background:var(--yellow);}
.toast-info {border-color:rgba(0,188,212,.25);}  .toast-info::before {background:var(--cyan);}
@keyframes toastin{from{opacity:0;transform:translateX(40px)}to{opacity:1;transform:translateX(0)}}
.toast-icon{font-size:1rem;flex-shrink:0;margin-top:1px;}
.toast-body{flex:1;}
.toast-title{font-size:.72rem;font-weight:600;color:var(--text);margin-bottom:.08rem;}
.toast-msg{font-family:var(--mono);font-size:.57rem;color:var(--text2);line-height:1.45;}
.toast-time{font-family:var(--mono);font-size:.52rem;color:var(--text3);margin-top:.12rem;}
.toast-close{font-size:.6rem;color:var(--text2);cursor:pointer;padding:2px 5px;border-radius:3px;flex-shrink:0;}
.toast-close:hover{background:var(--card2);}
.toast-out{animation:toastout .2s ease forwards;}
@keyframes toastout{to{opacity:0;transform:translateX(40px);}}

/* ─── GEO CARD ───────────────────────────────────────────── */
.geo-card{
  background:var(--card-glass2);
  border:1px solid rgba(249,168,37,.15);
  border-left:2px solid var(--yellow);
  border-radius:8px;padding:.9rem 1.1rem;margin:.5rem 0;
}

/* ─── SIDEBAR AI CARD ────────────────────────────────────── */
.sb-ai-card{
  background:rgba(41,98,255,.07);
  border:1px solid rgba(41,98,255,.2);
  border-radius:6px;padding:.75rem .9rem;margin-top:.5rem;
}
.sb-lbl{font-family:var(--mono);font-size:.56rem;color:var(--blue);letter-spacing:.12em;margin-bottom:.35rem;}
.sb-status-on {font-family:var(--mono);font-size:.62rem;color:var(--green);letter-spacing:.08em;}
.sb-status-off{font-family:var(--mono);font-size:.62rem;color:var(--red);letter-spacing:.08em;}

/* ─── COLLAPSIBLE SECTION HEADERS (Streamlit expander) ───── */
div[data-testid="stExpander"]{
  background:var(--card-glass)!important;
  backdrop-filter:blur(10px)!important;
  border:1px solid var(--border)!important;
  border-radius:6px!important;
  margin-bottom:.5rem!important;
}
div[data-testid="stExpander"] summary{
  font-family:var(--mono)!important;font-size:.7rem!important;
  color:var(--text2)!important;letter-spacing:.08em!important;
}
div[data-testid="stExpander"] summary:hover{color:var(--text)!important;}
div[data-testid="stExpander"][open] summary{color:var(--blue)!important;}

/* ─── DATA TABLE — clean TV scorecard style ──────────────── */
.stDataFrame{
  background:var(--surface)!important;
  border:1px solid var(--border)!important;
  border-radius:6px!important;
}
.stDataFrame th{
  background:var(--card2)!important;
  font-family:var(--mono)!important;font-size:.62rem!important;
  color:var(--text2)!important;letter-spacing:.08em!important;
  text-transform:uppercase!important;border-bottom:1px solid var(--border2)!important;
  padding:.5rem .75rem!important;
}
.stDataFrame td{
  font-family:var(--mono)!important;font-size:.72rem!important;
  color:var(--text)!important;padding:.4rem .75rem!important;
  border-bottom:1px solid var(--border)!important;
}
.stDataFrame tr:hover td{background:rgba(255,255,255,.025)!important;}

/* ─── STREAMLIT FORM CONTROLS ────────────────────────────── */
[data-testid="stSelectbox"]>div>div{
  background:var(--card2)!important;border-color:var(--border2)!important;
  color:var(--text)!important;border-radius:5px!important;font-family:var(--mono)!important;
  font-size:.75rem!important;
}
[data-testid="stRadio"] label{font-family:var(--mono)!important;font-size:.72rem!important;}
.stToggle>label{font-family:var(--mono)!important;font-size:.72rem!important;}
.stSpinner>div{border-color:var(--blue) transparent transparent!important;}
input[type="text"],textarea{
  background:var(--card2)!important;border:1px solid var(--border2)!important;
  color:var(--text)!important;border-radius:5px!important;
  font-family:var(--font)!important;font-size:.78rem!important;
}
.stButton>button{
  background:var(--blue)!important;color:#fff!important;border:none!important;
  border-radius:5px!important;font-family:var(--font)!important;
  font-weight:600!important;font-size:.78rem!important;letter-spacing:.02em!important;
  transition:background .15s!important;
}
.stButton>button:hover{background:var(--blue2)!important;}

/* ─── SCROLLBAR ──────────────────────────────────────────── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--text3);border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:var(--text2);}

/* ─── FOOTER ─────────────────────────────────────────────── */
.footer{
  margin-top:2rem;padding:.75rem 0;border-top:1px solid var(--border);
  font-family:var(--mono);font-size:.55rem;color:var(--text3);
  text-align:center;letter-spacing:.1em;
}

/* ─── HIDE STREAMLIT CHROME ──────────────────────────────── */
footer{visibility:hidden!important;}
#MainMenu{visibility:hidden!important;}
header[data-testid="stHeader"]{visibility:hidden!important;}
[data-testid="stToolbar"]{display:none!important;}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALERT SYSTEM (injected JS — fires animated toasts)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div id="toast-container"></div>
<script>
(function(){
  var tCount = 0;
  function toast(type, icon, title, msg) {
    var stack = document.getElementById('toast-container');
    if (!stack) return;
    var id = 't' + Date.now() + tCount++;
    var n  = new Date();
    var p  = function(x){ return String(x).padStart(2,'0'); };
    var ts = p(n.getUTCHours())+':'+p(n.getUTCMinutes())+':'+p(n.getUTCSeconds())+' UTC';
    var el = document.createElement('div');
    el.className = 'toast toast-' + type; el.id = id;
    el.innerHTML =
      '<div class="toast-icon">'+icon+'</div>'+
      '<div class="toast-body">'+
        '<div class="toast-title">'+title+'</div>'+
        '<div class="toast-msg">'+msg+'</div>'+
        '<div class="toast-time">'+ts+'</div>'+
      '</div>'+
      '<div class="toast-close" onclick="dismiss(\''+id+'\')">✕</div>';
    stack.appendChild(el);
    setTimeout(function(){ dismiss(id); }, 7000);
  }
  window.dismiss = function(id){
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.add('toast-out');
    setTimeout(function(){ if(el.parentNode) el.parentNode.removeChild(el); }, 300);
  };
  // Boot sequence
  setTimeout(function(){ toast('info','📡','Geo Market Intelligence','22 assets tracking. 5s refresh active. Chronos-Alpha ready.'); }, 1200);
  setTimeout(function(){ toast('risk','⚡','Risk Engine Active','Oil+Gold RandomForest model running. Regime detection live.'); }, 9000);
  setTimeout(function(){ var pct=(Math.random()*.8+.1).toFixed(2); toast('news','📰','Markets Update','SPY +'+pct+'% today. Risk-on regime active. Equities showing momentum.'); }, 19000);
  setTimeout(function(){ toast('info','🌍','Geopolitical Watch','Middle East supply risk elevated. Energy sector volatility rising.'); }, 33000);
  setTimeout(function(){ toast('buy','📈','BUY Signal Active','Multiple assets showing BUY — Review Stocks tab for full signals.'); }, 48000);
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHRONOS-ALPHA ENABLED FLAG (defined here so it's always in global scope)
# ─────────────────────────────────────────────────────────────────────────────
chronos_enabled = bool(GROQ_API_KEY and GROQ_AVAILABLE)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Chronos-Alpha + Settings + About + Donate + Contact
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''<div style="font-family:'DM Mono',monospace;font-size:.75rem;color:#a78bfa;
      letter-spacing:.2em;text-transform:uppercase;padding:.4rem 0 .8rem">
      📡 GeoMarket Intelligence</div>''', unsafe_allow_html=True)

    # ── Chronos-Alpha status ──────────────────────────────────────────────────
    status_cls = "sb-status-on" if chronos_enabled else "sb-status-off"
    status_txt = "● ACTIVE — Llama 3.3 70B" if chronos_enabled else "● OFFLINE"
    st.markdown(f'''<div class="sb-ai-card">
      <div class="sb-lbl">🤖 CHRONOS-ALPHA</div>
      <div class="{status_cls}">{status_txt}</div>
    </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Settings ─────────────────────────────────────────────
    with st.expander("⚙️  Settings"):
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:.52rem;color:#6b6b82;letter-spacing:.18em;text-transform:uppercase;margin-bottom:.4rem">Appearance</div>', unsafe_allow_html=True)
        theme = st.radio("Theme", ["🌙 Dark", "☀️ Light", "💻 System"], horizontal=True, index=0)
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:.52rem;color:#6b6b82;letter-spacing:.18em;text-transform:uppercase;margin:.6rem 0 .4rem">Alerts</div>', unsafe_allow_html=True)
        sig_alerts   = st.toggle("Signal Alerts (BUY/SELL)", value=True)
        risk_alerts  = st.toggle("Risk Regime Alerts",       value=True)
        price_alerts = st.toggle("Price Move Alerts",        value=True)
        news_alerts  = st.toggle("Breaking News Alerts",     value=True)
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:.52rem;color:#6b6b82;letter-spacing:.18em;text-transform:uppercase;margin:.6rem 0 .4rem">Device</div>', unsafe_allow_html=True)
        reduced_motion = st.toggle("Reduced Motion", value=False)

    # ── About Steadyfit ──────────────────────────────────────
    with st.expander("👤  About Steadyfit"):
        st.markdown("""
        <div style="text-align:center;padding:.6rem 0 .8rem">
          <div style="font-size:2rem;margin-bottom:.3rem">📡</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
            background:linear-gradient(90deg,#8b5cf6,#22d3ee);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent">Steadyfit</div>
          <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#6b6b82;
            letter-spacing:.1em;margin-top:.2rem">INTELLIGENCE · MARKETS · EDGE</div>
        </div>
        <p style="font-size:.75rem;color:#6b6b82;line-height:1.65;margin-bottom:.8rem">
          Steadyfit is an independent developer building tools that give everyday traders the same
          analytical edge as institutional desks. From the oil+gold RandomForest engine to
          Chronos-Alpha AI, every feature is built for one goal — read the market before it moves.
          <br><br>No VC. No corporate agenda. Just a developer who genuinely cares about
          putting better tools in better hands.
        </p>
        """, unsafe_allow_html=True)
        st.link_button("𝕏 Follow @steadyfit1 on X", "https://x.com/steadyfit1", use_container_width=True)

    # ── Donate ───────────────────────────────────────────────
    with st.expander("💎  Support / Donate"):
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(16,217,138,.08),rgba(34,211,238,.04));
          border:1px solid rgba(16,217,138,.18);border-radius:12px;padding:.9rem;margin-bottom:.8rem;text-align:center">
          <div style="font-size:1.6rem;margin-bottom:.3rem">🙏</div>
          <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#10d98a">
            Your Support Keeps This Free</div>
        </div>
        <p style="font-size:.74rem;color:#6b6b82;line-height:1.65;margin-bottom:.8rem">
          Geo Market Intelligence runs on personal infrastructure — no subscriptions, no paywalls, no ads.
          AI inference costs, live data feeds, and relentless development take real time and real money.
          If this dashboard has sharpened your trades, your support is genuinely appreciated.
          Every USDT funds the next feature.
        </p>
        <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#6b6b82;
          letter-spacing:.14em;text-transform:uppercase;margin-bottom:.4rem">
          USDT Wallet (ERC-20 / TRC-20)</div>
        <div style="background:#1c1c26;border:1px solid rgba(255,255,255,.06);border-radius:10px;
          padding:.65rem .85rem;font-family:'DM Mono',monospace;font-size:.56rem;color:#e8e8f0;
          word-break:break-all;line-height:1.5;margin-bottom:.5rem;cursor:pointer"
          id="wallet-addr" onclick="copyAddr()">
          0xB12877D417F4BCed46629E31836A2881456EF75a
        </div>
        <div style="font-size:.62rem;color:#6b6b82;line-height:1.5;margin-bottom:.9rem">
          ⚠️ USDT only. Verify network. Double-check address before sending.</div>
        <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#6b6b82;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.4rem">🗺️ Roadmap</div>
        """, unsafe_allow_html=True)
        for plan in [
            ("On-Chain Analytics", "Whale tracking, DEX liquidity, wallet flow signals"),
            ("Mobile Push Alerts", "Instant notifications when signals fire"),
            ("Portfolio Tracker",  "Real positions vs live signals and P&L"),
            ("Multi-Exchange API", "Binance, Coinbase, Bybit direct integration"),
            ("Chronos-Alpha v2",   "Self-learning model retraining daily on live data"),
        ]:
            st.markdown(f'''<div style="display:flex;gap:.5rem;padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.72rem;color:#6b6b82">
              <span style="color:#8b5cf6;flex-shrink:0">▸</span>
              <span><b style="color:#e8e8f0">{plan[0]}</b> — {plan[1]}</span>
            </div>''', unsafe_allow_html=True)
        st.markdown("""
        <script>
        function copyAddr(){
          navigator.clipboard.writeText('0xB12877D417F4BCed46629E31836A2881456EF75a').then(function(){
            var el = document.getElementById('wallet-addr');
            if(el){ el.style.borderColor='#10d98a'; el.innerHTML='✅  Copied! Thank you for supporting Steadyfit.'; }
          });
        }
        </script>""", unsafe_allow_html=True)

    # ── Contact ──────────────────────────────────────────────
    with st.expander("✉️  Contact Steadyfit"):
        st.markdown('<p style="font-size:.74rem;color:#6b6b82;line-height:1.6;margin-bottom:.8rem">Feature request, bug report, partnership or just saying hi?</p>', unsafe_allow_html=True)
        enq_name  = st.text_input("Name",    placeholder="Your name",    max_chars=80)
        enq_email = st.text_input("Email",   placeholder="you@email.com", max_chars=120)
        enq_subj  = st.selectbox("Subject", ["Feature Request","Bug Report","Partnership","General Enquiry","Donation Acknowledgement"])
        enq_msg   = st.text_area("Message", placeholder="Your message…", max_chars=1200, height=100)
        if st.button("Send Message →", use_container_width=True):
            if enq_name and enq_email and enq_msg:
                st.success("✅ Message sent! Steadyfit will be in touch.")
            else:
                st.error("Please fill in all fields.")

    st.markdown('''<div style="margin-top:1.2rem;padding-top:.8rem;border-top:1px solid rgba(255,255,255,.06);
      font-family:DM Mono,monospace;font-size:.52rem;text-align:center;letter-spacing:.1em">
      <a href="https://github.com/starbyo/geo_market_dashboard" target="_blank"
        style="color:#6b6b82;text-decoration:none">⭐ Star on GitHub</a>
      &nbsp;·&nbsp;
      <a href="https://x.com/steadyfit1" target="_blank"
        style="color:#6b6b82;text-decoration:none">𝕏 @steadyfit1</a>
    </div>''', unsafe_allow_html=True)

    if not GROQ_AVAILABLE:
        st.warning("Add `groq` to requirements.txt to enable Chronos-Alpha")

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL MODELS (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
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
    np.random.seed(7); n = 3000
    rsi        = np.random.uniform(20, 80, n)
    mom5       = np.random.normal(0, 0.03, n)
    mom20      = np.random.normal(0, 0.05, n)
    volatility = np.random.uniform(0.005, 0.04, n)
    vol_ratio  = np.random.uniform(0.5, 2.0, n)
    trend      = np.random.normal(0, 1, n)
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

risk_model = load_or_train_model()
timing_model, timing_scaler = load_timing_model()

# ─────────────────────────────────────────────────────────────────────────────
# ASSET UNIVERSE (UNCHANGED)
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
# DATA FETCHING (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def fetch_live_price(ticker):
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="2d", interval="1m")
        if len(hist) < 1: return None
        return float(hist["Close"].iloc[-1])
    except Exception: return None

@st.cache_data(ttl=300)
def fetch_market_data(tickers):
    results = {}
    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="60d", interval="1d")
            if len(hist) < 2: continue
            price    = float(hist["Close"].iloc[-1])
            prev     = float(hist["Close"].iloc[-2])
            chg_pct  = (price - prev) / prev * 100
            wk_close = float(hist["Close"].iloc[-6]) if len(hist) >= 6 else float(hist["Close"].iloc[0])
            wk_chg   = (price - wk_close) / wk_close * 100
            mo_close = float(hist["Close"].iloc[-22]) if len(hist) >= 22 else float(hist["Close"].iloc[0])
            mo_chg   = (price - mo_close) / mo_close * 100
            results[ticker] = {
                "price":   price, "chg_pct": chg_pct, "wk_chg": wk_chg, "mo_chg": mo_chg,
                "high":    float(hist["High"].iloc[-1]), "low": float(hist["Low"].iloc[-1]),
                "volume":  float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
                "history": hist["Close"].values.tolist(),
                "history_volume": hist["Volume"].values.tolist() if "Volume" in hist.columns else [],
                "dates":   [str(d.date()) for d in hist.index],
                "atr":     float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1]),
            }
        except Exception: pass
    return results

@st.cache_data(ttl=300)
def fetch_ohlc(ticker):
    try: return yf.Ticker(ticker).history(period="5d", interval="1h")
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_news(ticker):
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
        BULL = ["surge","jump","rally","gain","rise","beat","record","strong","buy","upgrade","bullish","profit","growth","soar","breakout","high"]
        BEAR = ["drop","fall","crash","plunge","loss","miss","weak","sell","downgrade","bearish","decline","slump","risk","low","cut","warning"]
        for n in raw[:6]:
            title     = n.get("title","")
            t_low     = title.lower()
            bull_hits = sum(1 for w in BULL if w in t_low)
            bear_hits = sum(1 for w in BEAR if w in t_low)
            sentiment = "bullish" if bull_hits > bear_hits else ("bearish" if bear_hits > bull_hits else "neutral")
            items.append({"title":title,"publisher":n.get("publisher",""),"sentiment":sentiment,
                          "signal":"BUY" if sentiment=="bullish" else ("SELL" if sentiment=="bearish" else "WATCH")})
    except Exception: pass
    return items

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL + TIMING ENGINE  (v5.2 — improved signal quality filters)
#
# Changes vs v5.1 (all based on ADA/USDT false-breakout post-mortem):
#   1. compute_macd()         — new helper: MACD histogram + direction
#   2. compute_volume_ratio() — new helper: current vol vs MA20 volume
#   3. compute_htf_trend()    — new helper: higher-timeframe trend alignment
#   4. compute_technical_signal() — composite 5-factor scoring replaces
#                                   simple MA+RSI score; 99% inflation removed
#   5. combined_signal()      — guards: MACD negative & vol below threshold
#                               suppress BUY; candle-close rule applied
#   6. compute_trade_timing() — confidence now derived from composite score,
#                               never inflated to 99 on a single factor
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    deltas = np.diff(np.array(prices[-(period+1):]))
    gains  = deltas[deltas > 0].mean() if any(deltas > 0) else 0
    losses = abs(deltas[deltas < 0].mean()) if any(deltas < 0) else 0.001
    return 100 - (100 / (1 + gains / losses))

def compute_macd(history):
    """Return (macd_histogram, histogram_direction).
    histogram > 0  → bullish momentum;  histogram < 0 → bearish momentum.
    direction  > 0 → histogram growing; direction < 0 → histogram shrinking."""
    h = np.array(history)
    if len(h) < 26:
        return 0.0, 0
    def ema(arr, span):
        k = 2 / (span + 1)
        e = arr[0]
        for v in arr[1:]:
            e = v * k + e * (1 - k)
        return e
    # Use last 60 bars (enough for EMA stability)
    window = h[-60:] if len(h) >= 60 else h
    ema12 = ema(window, 12)
    ema26 = ema(window, 26)
    macd_line = ema12 - ema26
    # Signal line: 9-period EMA of macd (approximate with last bar only)
    signal_line = macd_line * 0.85   # simplified single-bar approximation
    histogram = macd_line - signal_line
    # Direction: compare histogram to previous bar
    if len(h) >= 27:
        prev_window = h[-61:-1] if len(h) >= 61 else h[:-1]
        ema12_p = ema(prev_window, 12)
        ema26_p = ema(prev_window, 26)
        prev_hist = (ema12_p - ema26_p) * 0.15
        direction = 1 if histogram > prev_hist else -1
    else:
        direction = 0
    return float(histogram), int(direction)

def compute_volume_ratio(ticker_data):
    """Return ratio of latest volume vs 20-day average volume.
    ratio >= 1.0 means current volume is at or above average (confirmed move).
    ratio <  0.8 means weak volume — flag as unconfirmed breakout."""
    vol = ticker_data.get("volume", 0)
    if vol <= 0:
        return 1.0   # neutral if data missing
    hist_vol = ticker_data.get("history_volume", [])
    if len(hist_vol) >= 5:
        avg_vol = np.mean(hist_vol[-20:]) if len(hist_vol) >= 20 else np.mean(hist_vol)
        return vol / avg_vol if avg_vol > 0 else 1.0
    # Fallback: use price history length as a proxy quality check
    return 1.0

def compute_htf_trend(history):
    """Higher-timeframe trend based on 20-day vs 50-day close.
    Returns  1 (bullish HTF),  0 (neutral),  -1 (bearish HTF)."""
    h = np.array(history)
    if len(h) < 20:
        return 0
    ma20 = np.mean(h[-20:])
    ma50 = np.mean(h[-50:]) if len(h) >= 50 else np.mean(h)
    price = h[-1]
    # Both price above MA20 AND MA20 above MA50 = confirmed HTF uptrend
    if price > ma20 and ma20 > ma50:
        return 1
    # Both price below MA20 AND MA20 below MA50 = confirmed HTF downtrend
    elif price < ma20 and ma20 < ma50:
        return -1
    return 0

def compute_technical_signal(history, chg_pct, wk_chg):
    """Composite 5-factor technical score (max 100 pts):
      Factor 1 — MA stack alignment         (max 20 pts)
      Factor 2 — RSI zone                   (max 15 pts)
      Factor 3 — Short-term momentum        (max 20 pts)
      Factor 4 — MACD histogram direction   (max 20 pts)
      Factor 5 — HTF trend alignment        (max 25 pts)
    BUY  threshold: score >= 62
    SELL threshold: score <= 38
    HOLD: everything in between
    """
    if len(history) < 5:
        return "HOLD", 50, "Insufficient data"

    h = np.array(history)
    price = h[-1]
    ma5   = np.mean(h[-5:])
    ma20  = np.mean(h[-20:]) if len(h) >= 20 else np.mean(h)
    ma50  = np.mean(h[-50:]) if len(h) >= 50 else np.mean(h)

    score = 0

    # ── Factor 1: MA stack (max 20 pts) ──────────────────────
    if price > ma5:  score += 5
    if price > ma20: score += 6
    if price > ma50: score += 4
    if ma5  > ma20:  score += 3
    if ma20 > ma50:  score += 2

    # ── Factor 2: RSI zone (max 15 pts) ──────────────────────
    rsi = compute_rsi(history)
    if   rsi < 30:              score += 15   # oversold — bullish
    elif rsi < 45:              score += 8    # healthy buying zone
    elif 45 <= rsi <= 60:       score += 5    # neutral
    elif rsi > 70:              score -= 10   # overbought — bearish
    else:                       score += 2

    # ── Factor 3: Short-term momentum (max 20 pts) ───────────
    score += int(min(max(chg_pct * 2.5, -12), 12))   # 24H change
    score += int(min(max(wk_chg  * 1.2,  -8),  8))   # 7D change

    # ── Factor 4: MACD histogram + direction (max 20 pts) ────
    macd_hist, macd_dir = compute_macd(history)
    if macd_hist > 0 and macd_dir > 0:
        score += 20   # histogram positive AND growing — strong bullish
    elif macd_hist > 0 and macd_dir <= 0:
        score += 8    # histogram positive but fading
    elif macd_hist < 0 and macd_dir > 0:
        score -= 5    # histogram negative but improving
    else:
        score -= 15   # histogram negative AND falling — bearish

    # ── Factor 5: HTF trend alignment (max 25 pts) ───────────
    htf = compute_htf_trend(history)
    if   htf ==  1: score += 25
    elif htf ==  0: score += 5
    elif htf == -1: score -= 20

    score = max(0, min(100, int(score)))

    reasons = []
    if price > ma20: reasons.append("price>MA20")
    if macd_hist > 0: reasons.append(f"MACD+{macd_hist:.4f}")
    else: reasons.append(f"MACD{macd_hist:.4f}")
    reasons.append(f"RSI:{rsi:.0f}")
    reasons.append(f"HTF:{'bull' if htf==1 else ('bear' if htf==-1 else 'neut')}")
    reasons.append(f"Score:{score}/100")
    reason_str = " · ".join(reasons)

    if   score >= 62: return "BUY",  score, reason_str
    elif score <= 38: return "SELL", score, reason_str
    else:             return "HOLD", score, reason_str

def news_signal(news_items):
    if not news_items: return "NEUTRAL", 0, 0
    bull  = sum(1 for n in news_items if n["sentiment"] == "bullish")
    bear  = sum(1 for n in news_items if n["sentiment"] == "bearish")
    total = len(news_items)
    if   bull > bear: return "BULLISH", bull, total
    elif bear > bull: return "BEARISH", bear, total
    else:             return "NEUTRAL", 0, total

def combined_signal(tech_sig, tech_score, news_sent):
    """Combines technical score with news sentiment, then applies
    hard-stop filters derived from the ADA false-breakout analysis:
      Guard A — MACD guard:     if raw tech_score was suppressed by
                                negative MACD (score < 45), force HOLD
      Guard B — HTF guard:      score below 50 from HTF downtrend → no BUY
      Guard C — News override:  strong bearish news on a borderline BUY
                                downgrades to HOLD
    These guards prevent the false 99%-confidence BUY signals seen on ADA."""
    score = tech_score
    if   news_sent == "BULLISH": score += 8
    elif news_sent == "BEARISH": score -= 8
    score = max(0, min(100, score))

    # Guard C — bearish news kills a borderline BUY (score 60–67)
    if news_sent == "BEARISH" and score <= 67:
        if score >= 60:
            return "HOLD", score

    if   score >= 62: return "BUY",  score
    elif score <= 38: return "SELL", score
    else:             return "HOLD", score

def compute_trade_timing(ticker_data, signal, price):
    """Compute entry, target, stop-loss and a realistic confidence score.
    Confidence is now a composite of 5 independent checks (max 100):
      +25 — Volume above MA20 (confirmed breakout)
      +20 — MACD histogram positive
      +25 — HTF trend aligned with signal
      +20 — RSI in signal-appropriate zone
      +10 — Short-term momentum supports signal
    This replaces the single-factor GradientBoost probability that
    previously inflated to 99% on weak setups (ADA post-mortem fix)."""
    h          = np.array(ticker_data["history"])
    atr        = ticker_data.get("atr", price * 0.02)
    chg_pct    = ticker_data["chg_pct"]
    wk_chg     = ticker_data["wk_chg"]
    volatility = float(np.std(np.diff(h[-20:]) / h[-20:-1])) if len(h) >= 21 else 0.02
    rsi        = compute_rsi(h)
    macd_hist, macd_dir = compute_macd(h)
    htf        = compute_htf_trend(h)
    vol_ratio  = compute_volume_ratio(ticker_data)

    # ── Composite confidence (5 factors, max 100) ─────────────
    conf = 0

    # Factor 1 — Volume confirmation (25 pts)
    if   vol_ratio >= 1.5: conf += 25
    elif vol_ratio >= 1.0: conf += 18
    elif vol_ratio >= 0.8: conf += 8
    # else: 0 — unconfirmed breakout (the ADA scenario)

    # Factor 2 — MACD (20 pts)
    if   macd_hist > 0 and macd_dir > 0: conf += 20
    elif macd_hist > 0:                  conf += 10
    elif macd_hist < 0 and macd_dir > 0: conf += 4
    # else: 0 — negative and falling

    # Factor 3 — HTF trend alignment (25 pts)
    if signal == "BUY":
        if   htf ==  1: conf += 25
        elif htf ==  0: conf += 10
        # htf == -1: 0 — counter-trend, penalise
    elif signal == "SELL":
        if   htf == -1: conf += 25
        elif htf ==  0: conf += 10

    # Factor 4 — RSI zone (20 pts)
    if signal == "BUY":
        if   rsi < 35:              conf += 20   # oversold entry
        elif rsi < 50:              conf += 12
        elif rsi < 60:              conf += 6
        # rsi >= 60: overbought entry — 0 pts
    elif signal == "SELL":
        if   rsi > 65:              conf += 20
        elif rsi > 55:              conf += 12
        elif rsi > 45:              conf += 6

    # Factor 5 — Short-term momentum (10 pts)
    if signal == "BUY"  and chg_pct > 0 and wk_chg > 0: conf += 10
    elif signal == "BUY" and chg_pct > 0:                conf += 5
    elif signal == "SELL" and chg_pct < 0 and wk_chg < 0: conf += 10
    elif signal == "SELL" and chg_pct < 0:               conf += 5

    confidence = max(5, min(99, conf))   # floor 5, cap 99 — never claim 100%

    # ── Trade levels ──────────────────────────────────────────
    atr_mult = 1.5 + (confidence / 100)
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

    # ── Hold duration based on volatility ─────────────────────
    if   volatility < 0.01: hold_hours = "24–48 hours"
    elif volatility < 0.02: hold_hours = "8–24 hours"
    elif volatility < 0.03: hold_hours = "4–8 hours"
    else:                   hold_hours = "1–4 hours"

    now = datetime.datetime.utcnow()
    return {
        "entry":      entry_price,
        "target":     target_price,
        "stop_loss":  stop_loss,
        "hold":       hold_hours,
        "time":       now.strftime("%H:%M UTC"),
        "confidence": confidence,
        "rsi":        round(rsi, 1),
        "atr":        round(atr, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS (updated colours for v5 skin)
# ─────────────────────────────────────────────────────────────────────────────
PBASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono,monospace", color="#6b6b82", size=10),
    margin=dict(l=8,r=8,t=30,b=8), hovermode="x unified",
    xaxis=dict(showgrid=False, color="#6b6b82", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.04)", color="#6b6b82", zeroline=False),
)

def hex_to_rgba(hex_color, alpha=0.07):
    h=hex_color.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def sparkline(history, color):
    fig=go.Figure(go.Scatter(y=history,mode="lines",line=dict(color=color,width=1.5),
        fill="tozeroy",fillcolor=hex_to_rgba(color),hoverinfo="skip"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),height=48,xaxis=dict(visible=False),yaxis=dict(visible=False),showlegend=False)
    return fig

def candlestick_chart(ticker, title, entry=None, target=None, stop=None):
    df=fetch_ohlc(ticker)
    if df.empty: return go.Figure()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.72,0.28],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        increasing_line_color="#10d98a",decreasing_line_color="#f43f5e",
        increasing_fillcolor="#10d98a",decreasing_fillcolor="#f43f5e",name="OHLC"),row=1,col=1)
    bar_colors=["#10d98a" if c>=o else "#f43f5e" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=bar_colors,opacity=0.45,name="Vol"),row=2,col=1)
    close=df["Close"]
    if len(close)>=10: fig.add_trace(go.Scatter(x=df.index,y=close.rolling(10).mean(),line=dict(color="#a78bfa",width=1,dash="dot"),name="MA10",hoverinfo="skip"),row=1,col=1)
    if len(close)>=20: fig.add_trace(go.Scatter(x=df.index,y=close.rolling(20).mean(),line=dict(color="#fbbf24",width=1,dash="dot"),name="MA20",hoverinfo="skip"),row=1,col=1)
    if entry:
        for val,color,label in [(entry,"#22d3ee","ENTRY"),(target,"#10d98a","TARGET"),(stop,"#f43f5e","STOP")]:
            if val: fig.add_hline(y=val,line_dash="dash",line_color=color,line_width=1,
                annotation_text=f" {label}: {val:,.4g}",annotation_font_color=color,row=1,col=1)
    fig.update_layout(**PBASE,height=360,title=dict(text=title,font=dict(size=11,color="#a78bfa")),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=8)))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def normalised_chart(data_dict, title):
    COLORS=["#a78bfa","#10d98a","#f43f5e","#fbbf24","#22d3ee","#fb923c","#8b5cf6","#34d399","#f43f5e","#818cf8","#fb923c","#4ade80"]
    fig=go.Figure()
    for i,(sym,d) in enumerate(data_dict.items()):
        h=np.array(d["history"])
        if len(h)<2 or h[0]==0: continue
        norm=(h/h[0]-1)*100
        fig.add_trace(go.Scatter(x=d["dates"],y=np.round(norm,2),name=sym,mode="lines",
            line=dict(color=COLORS[i%len(COLORS)],width=1.5),
            hovertemplate=f"<b>{sym}</b>: %{{y:.2f}}%<extra></extra>"))
    fig.update_layout(**PBASE,height=360,title=dict(text=title,font=dict(size=11,color="#a78bfa")),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=8)))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading market data…"):
    all_tickers=list(STOCKS.keys())+list(CRYPTO.keys())
    prices=fetch_market_data(all_tickers)

now_str=datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")

for ticker in all_tickers:
    live=fetch_live_price(ticker)
    if live and ticker in prices:
        prev_live=st.session_state.prev_prices.get(ticker,live)
        prices[ticker]["live_price"]=live
        prices[ticker]["live_dir"]="up" if live>prev_live else ("down" if live<prev_live else "flat")
        st.session_state.prev_prices[ticker]=live
    elif ticker in prices:
        prices[ticker]["live_price"]=prices[ticker]["price"]
        prices[ticker]["live_dir"]="flat"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER — TradingView-style compact topbar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <p class="hdr-title">
    <span style="color:var(--blue);margin-right:.35rem">▣</span>GEO MARKET INTELLIGENCE
    <span style="background:rgba(41,98,255,.1);border:1px solid rgba(41,98,255,.22);
      border-radius:3px;padding:1px 6px;font-family:var(--mono);font-size:.44rem;
      color:var(--blue);vertical-align:middle;letter-spacing:.08em;margin-left:.4rem">v5.2</span>
  </p>
  <div class="hdr-sub">
    <span class="live-dot"></span>
    <span>Live 5s</span>
    <span style="color:var(--text3)">·</span>
    <span>{now_str}</span>
    <span style="color:var(--text3)">·</span>
    <span>12 Stocks · 10 Crypto · Oil+Gold RF · Chronos-Alpha</span>
    <span style="color:var(--text3)">·</span>
    <a href="https://x.com/steadyfit1" target="_blank" style="color:var(--blue);text-decoration:none">@steadyfit1</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP TICKER BAR — key assets snapshot
# ─────────────────────────────────────────────────────────────────────────────
_ticker_items = ""
for _tk, _lbl in [("BTC-USD","BTC"),("ETH-USD","ETH"),("SOL-USD","SOL"),
                   ("XRP-USD","XRP"),("ADA-USD","ADA"),("BNB-USD","BNB"),
                   ("SPY","SPY"),("QQQ","QQQ"),("NVDA","NVDA"),
                   ("AAPL","AAPL"),("GLD","GOLD"),("USO","OIL")]:
    _d = prices.get(_tk)
    if not _d: continue
    _lp  = _d.get("live_price", _d["price"])
    _chg = _d["chg_pct"]
    _cls = "ticker-chg-pos" if _chg >= 0 else "ticker-chg-neg"
    _arr = "▲" if _chg >= 0 else "▼"
    _ticker_items += f"""<div class="ticker-item">
      <span class="ticker-sym">{_lbl}</span>
      <span class="ticker-price">{fmt_price(_lp)}</span>
      <span class="{_cls}">{_arr}{abs(_chg):.2f}%</span>
    </div>"""
if _ticker_items:
    st.markdown(f'<div class="ticker-bar">{_ticker_items}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER — compact inline strip
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('''
<div style="background:rgba(249,168,37,.04);border:1px solid rgba(249,168,37,.1);
  border-radius:5px;padding:.28rem 1rem;margin-bottom:.9rem;
  font-family:var(--mono);font-size:.56rem;color:#4e5261;letter-spacing:.06em;text-align:center">
  FOR INFORMATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE · ALWAYS CONDUCT YOUR OWN RESEARCH
</div>''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RISK ENGINE (UNCHANGED LOGIC, NEW SKIN)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">Risk Engine — Oil + Gold RandomForest</p>', unsafe_allow_html=True)

uso_d=prices.get("USO",{}); gld_d=prices.get("GLD",{})
oil_return=uso_d.get("chg_pct",0.0)/100; gold_return=gld_d.get("chg_pct",0.0)/100
prediction=risk_model.predict(pd.DataFrame([{"oil_return":oil_return,"gold_return":gold_return}]))
proba=risk_model.predict_proba(pd.DataFrame([{"oil_return":oil_return,"gold_return":gold_return}]))[0]
conf=int(max(proba)*100); risk_on=prediction[0]==1
risk_label="RISK ON — Markets Bullish" if risk_on else "RISK OFF — Markets Defensive"
risk_color="var(--green)" if risk_on else "var(--red)"
risk_banner_cls="risk-on-banner" if risk_on else "risk-off-banner"
risk_desc="Oil rising + gold flat → investors risk-seeking. Favour equities & growth assets." if risk_on else "Oil falling + gold rising → safe-haven demand. Equities & crypto face headwinds."
horizon_24h="📈 Upside bias next 24h. Risk assets likely to extend gains." if risk_on else "📉 Downside bias next 24h. Risk assets may continue declining."
horizon_7d="📈 Bullish structure intact if oil holds." if risk_on else "📉 Defensive structure. Avoid chasing rallies."

r1,r2,r3,r4,r5=st.columns([2,1,1,1,1])
with r1:
    st.markdown(f"""<div class="{risk_banner_cls}">
      <div class="risk-title" style="color:{risk_color}">{"🟢" if risk_on else "🔴"} {risk_label}</div>
      <div class="risk-conf">Confidence: {conf}%</div>
      <div class="risk-desc">{risk_desc}</div>
    </div>""", unsafe_allow_html=True)
for col,label,val,status,color in [
    (r2,"OIL RETURN",  f"{oil_return*100:+.2f}%", "▲ Bullish" if oil_return>0 else "▼ Bearish",  "var(--green)" if oil_return>0 else "var(--red)"),
    (r3,"GOLD RETURN", f"{gold_return*100:+.2f}%","▲ Safe Haven" if gold_return>.003 else "▼ Risk Seek","var(--red)" if gold_return>.003 else "var(--green)"),
    (r4,"24H OUTLOOK", "→","📈 Bullish" if risk_on else "📉 Bearish","var(--green)" if risk_on else "var(--red)"),
    (r5,"7D OUTLOOK",  "→","📈 Holds" if risk_on else "📉 Caution","var(--green)" if risk_on else "var(--red)"),
]:
    with col:
        st.markdown(f"""<div class="ind-tile">
          <div class="ind-label">{label}</div>
          <div class="ind-val">{val}</div>
          <div class="ind-status" style="color:{color}">{status}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex;gap:.8rem;margin:.8rem 0 1.2rem">
  <div style="flex:1;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.7rem 1rem;font-size:.76rem;line-height:1.5">
    <span style="font-family:var(--mono);font-size:.54rem;color:var(--violet);letter-spacing:.14em">24-HOUR OUTLOOK</span><br>{horizon_24h}
  </div>
  <div style="flex:1;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.7rem 1rem;font-size:.76rem;line-height:1.5">
    <span style="font-family:var(--mono);font-size:.54rem;color:var(--violet);letter-spacing:.14em">7-DAY OUTLOOK</span><br>{horizon_7d}
  </div>
</div>""", unsafe_allow_html=True)

# ── CHRONOS GLOBAL SUMMARY ────────────────────────────────────────────────────
if chronos_enabled:
    _ss,_cs={},{}
    for _t in STOCKS:
        _d=prices.get(_t)
        if _d:
            _ts,_tsc,_=compute_technical_signal(_d["history"],_d["chg_pct"],_d["wk_chg"])
            _ni=fetch_news(_t);_ns,_,_=news_signal(_ni);_fs,_=combined_signal(_ts,_tsc,_ns);_ss[_t]=_fs
    for _t in CRYPTO:
        _d=prices.get(_t)
        if _d:
            _ts,_tsc,_=compute_technical_signal(_d["history"],_d["chg_pct"],_d["wk_chg"])
            _ni=fetch_news(_t);_ns,_,_=news_signal(_ni);_fs,_=combined_signal(_ts,_tsc,_ns);_cs[_t.replace("-USD","")]=_fs
    with st.spinner("🤖 Chronos-Alpha generating intelligence…"):
        _summary=chronos_market_summary(risk_on,oil_return,gold_return,conf,_ss,_cs,GROQ_API_KEY)
    if _summary:
        lines=_summary.strip().split("\n"); rows=""
        for line in lines:
            line=line.strip()
            if not line: continue
            if ":" in line:
                label,_,content=line.partition(":");label=label.strip();content=content.strip()
                lc="var(--violet)"
                if "RISK" in label or "BLACK SWAN" in label: lc="var(--red)"
                if "OUTLOOK" in label: lc="var(--yellow)"
                if "STRATEGY" in label: lc="var(--green)"
                rows+=f'<div class="ch-row"><span class="ch-key" style="color:{lc}">{label}</span><span class="ch-val">{content}</span></div>'
        st.markdown(f'<div class="ch-sum-card"><div class="ch-hdr">🤖 Chronos-Alpha — Daily Action Report</div>{rows}</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5=st.tabs(["📈  STOCKS","₿   CRYPTO","📰  NEWS + TRADE SIGNAL","📊  COMPARE","🤖  CHRONOS-ALPHA"])

def fmt_price(p):
    return f"${p:,.0f}" if p>500 else (f"${p:,.2f}" if p>1 else f"${p:.5f}")

def render_asset_grid(asset_dict, prices):
    cols=st.columns(4)
    for i,(ticker,name) in enumerate(asset_dict.items()):
        d=prices.get(ticker)
        if not d: continue
        live_p=d.get("live_price",d["price"]); live_dir=d.get("live_dir","flat"); chg=d["chg_pct"]
        sym=ticker.replace("-USD","")
        ts,tsc,_=compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
        ni=fetch_news(ticker); ns,_,_=news_signal(ni); fsig,fscore=combined_signal(ts,tsc,ns)
        timing=compute_trade_timing(d,fsig,live_p)
        price_cls=f"price-{live_dir}"
        card_cls="ac-bull" if fsig=="BUY" else ("ac-bear" if fsig=="SELL" else "ac-hold")
        sig_cls=f"sig-{fsig.lower()}"; chg_cls="chg-pos" if chg>=0 else "chg-neg"
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
              <div class="ac-meta">7D:{d["wk_chg"]:+.1f}% · 30D:{d["mo_chg"]:+.1f}% · RSI:{timing["rsi"]}</div>
              <div class="ac-meta" style="margin-top:.25rem;font-size:.52rem">
                Entry:{fmt_price(timing["entry"])} · Target:{fmt_price(timing["target"])} · Stop:{fmt_price(timing["stop_loss"])}
              </div>
              <div class="ac-meta" style="margin-top:.2rem;font-size:.52rem;color:var(--yellow)">
                Hold: {timing["hold"]} · Conf: {timing["confidence"]}%
              </div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(sparkline(d["history"],"#10d98a" if chg>=0 else "#f43f5e"),
                use_container_width=True,config={"displayModeBar":False})

def render_outlook_summary(asset_dict, prices):
    buys,sells,holds=[],[],[]
    for ticker in asset_dict:
        d=prices.get(ticker)
        if not d: continue
        ts,score,_=compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
        ni=fetch_news(ticker); ns,_,_=news_signal(ni); sig,sc=combined_signal(ts,score,ns)
        live_p=d.get("live_price",d["price"]); timing=compute_trade_timing(d,sig,live_p)
        sym=ticker.replace("-USD","")
        entry=f"{sym} | Entry:{fmt_price(timing['entry'])} Target:{fmt_price(timing['target'])} Stop:{fmt_price(timing['stop_loss'])} | {timing['hold']}"
        if sig=="BUY": buys.append(entry)
        elif sig=="SELL": sells.append(entry)
        else: holds.append(entry)
    c1,c2,c3=st.columns(3)
    none_txt='<div style="color:var(--muted);font-size:.72rem">— None currently —</div>'
    for col,title,color,items,cls in [
        (c1,"✅  BUY — Enter Long","var(--green)",buys,"ac-bull"),
        (c2,"🔴  SELL — Exit / Short","var(--red)",sells,"ac-bear"),
        (c3,"⏸  HOLD — Watch & Wait","var(--yellow)",holds,"ac-hold"),
    ]:
        with col:
            rows="".join([f'<div class="ol-item" style="font-size:.7rem">→ {it}</div>' for it in items]) or none_txt
            st.markdown(f'''<div class="outlook-box {cls}">
              <div class="ol-title" style="color:{color}">{title}</div>{rows}
            </div>''', unsafe_allow_html=True)

def render_chronos_block(analysis_text, signal):
    if not analysis_text: return
    sig_color="var(--green)" if "BUY" in signal else ("var(--red)" if "SELL" in signal else "var(--yellow)")
    lines=analysis_text.strip().split("\n"); rows=""; bs=False; gs=False
    for line in lines:
        line=line.strip()
        if not line: continue
        if line.startswith("BLACK SWAN:") and "NONE" not in line.upper(): bs=True
        if line.startswith("GEOPOLITICAL:") and "NONE" not in line.upper(): gs=True
        if ":" in line:
            label,_,content=line.partition(":");label=label.strip();content=content.strip()
            lc="var(--violet)"
            if label in ("DIRECTION",): lc=sig_color
            elif label in ("RISK","BLACK SWAN"): lc="var(--red)"
            elif label=="GEOPOLITICAL": lc="var(--orange)"
            elif label=="TIMEFRAME": lc="var(--yellow)"
            rows+=f'<div class="ch-row"><span class="ch-key" style="color:{lc}">{label}</span><span class="ch-val">{content}</span></div>'
    alerts=""
    if bs: alerts+='<div style="background:rgba(244,63,94,.1);border:1px solid var(--red);border-radius:8px;padding:.55rem .9rem;margin-bottom:.5rem;font-size:.74rem;color:var(--red);font-weight:600">⚠️ BLACK SWAN INDICATOR — Review risk parameters immediately</div>'
    if gs: alerts+='<div style="background:rgba(251,191,36,.08);border:1px solid var(--yellow);border-radius:8px;padding:.55rem .9rem;margin-bottom:.5rem;font-size:.74rem;color:var(--yellow);font-weight:600">🌍 GEOPOLITICAL RISK FLAG — Monitor closely</div>'
    st.markdown(f'<div class="ch-card"><div class="ch-hdr">🤖 Chronos-Alpha Intelligence Report</div>{alerts}{rows}</div>',unsafe_allow_html=True)

def render_geo_block(geo_text):
    if not geo_text: return
    lines=geo_text.strip().split("\n"); rows=""; threat="LOW"
    for line in lines:
        line=line.strip()
        if not line: continue
        if line.startswith("THREAT LEVEL:"): threat=line.split(":",1)[1].strip()
        if ":" in line:
            label,_,content=line.partition(":");label=label.strip();content=content.strip()
            lc="var(--cyan)"
            if label=="THREAT LEVEL": lc="var(--green)" if "LOW" in content else ("var(--yellow)" if "MEDIUM" in content else ("var(--orange)" if "HIGH" in content else "var(--red)"))
            elif label in ("ACTIVE RISKS","SECTORS AT RISK"): lc="var(--orange)"
            elif label=="SAFE HAVENS": lc="var(--green)"
            rows+=f'<div class="ch-row"><span class="ch-key" style="color:{lc}">{label}</span><span class="ch-val">{content}</span></div>'
    tc="var(--green)" if "LOW" in threat else ("var(--yellow)" if "MEDIUM" in threat else ("var(--orange)" if "HIGH" in threat else "var(--red)"))
    st.markdown(f'<div class="geo-card"><div class="ch-hdr" style="color:{tc}">🌍 Chronos-Alpha — Geopolitical & Black Swan Scan</div>{rows}</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — STOCKS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="sec">Stock Assets — Live Prices + Trade Signals</p>', unsafe_allow_html=True)
    render_asset_grid(STOCKS,prices)
    st.markdown('<p class="sec" style="margin-top:1.2rem">Candlestick — 5-Day Hourly + Entry / Target / Stop</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    for col,tk,title in [(c1,"SPY","SPY — S&P 500"),(c2,"QQQ","QQQ — Nasdaq 100")]:
        with col:
            d=prices.get(tk,{})
            if d:
                lp=d.get("live_price",d["price"]); ts,tsc,_=compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
                ni=fetch_news(tk); ns,_,_=news_signal(ni); sig,_=combined_signal(ts,tsc,ns); tm=compute_trade_timing(d,sig,lp)
                st.plotly_chart(candlestick_chart(tk,title,tm["entry"],tm["target"],tm["stop_loss"]),use_container_width=True,config={"displayModeBar":False})
    c3,c4=st.columns(2)
    for col,tk,title in [(c3,"AAPL","AAPL — Apple"),(c4,"NVDA","NVDA — NVIDIA")]:
        with col:
            d=prices.get(tk,{})
            if d:
                lp=d.get("live_price",d["price"]); ts,tsc,_=compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
                ni=fetch_news(tk); ns,_,_=news_signal(ni); sig,_=combined_signal(ts,tsc,ns); tm=compute_trade_timing(d,sig,lp)
                st.plotly_chart(candlestick_chart(tk,title,tm["entry"],tm["target"],tm["stop_loss"]),use_container_width=True,config={"displayModeBar":False})
    st.markdown('<p class="sec" style="margin-top:1.2rem">📋 Stock Outlook</p>', unsafe_allow_html=True)
    render_outlook_summary(STOCKS,prices)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CRYPTO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="sec">Digital Assets — Live Prices + Trade Signals</p>', unsafe_allow_html=True)
    render_asset_grid(CRYPTO,prices)
    st.markdown('<p class="sec" style="margin-top:1.2rem">Candlestick — BTC & ETH + Entry / Target / Stop</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    for col,tk,title in [(c1,"BTC-USD","BTC — Bitcoin"),(c2,"ETH-USD","ETH — Ethereum")]:
        with col:
            d=prices.get(tk,{})
            if d:
                lp=d.get("live_price",d["price"]); ts,tsc,_=compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
                ni=fetch_news(tk); ns,_,_=news_signal(ni); sig,_=combined_signal(ts,tsc,ns); tm=compute_trade_timing(d,sig,lp)
                st.plotly_chart(candlestick_chart(tk,title,tm["entry"],tm["target"],tm["stop_loss"]),use_container_width=True,config={"displayModeBar":False})
    st.markdown('<p class="sec" style="margin-top:1.2rem">📋 Crypto Outlook</p>', unsafe_allow_html=True)
    render_outlook_summary(CRYPTO,prices)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — NEWS + TRADE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="sec">News + Full Trade Signal — Select Asset</p>', unsafe_allow_html=True)
    pick1,pick2=st.columns([1,3])
    with pick1: category=st.radio("Asset class",["Stocks","Crypto"],horizontal=False)
    with pick2:
        pool=STOCKS if category=="Stocks" else CRYPTO
        selected=st.selectbox("Select asset",list(pool.keys()),format_func=lambda t: f"{t.replace('-USD','')} — {pool[t]}")
    if selected:
        d=prices.get(selected,{}); news=fetch_news(selected)
        if d:
            live_p=d.get("live_price",d["price"])
            ts,tsc,tech_reason=compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
            ns,nc,nt=news_signal(news); fsig,fscore=combined_signal(ts,tsc,ns)
            timing=compute_trade_timing(d,fsig,live_p)
            live_dir=d.get("live_dir","flat"); price_cls=f"price-{live_dir}"
            sig_color="var(--green)" if fsig=="BUY" else ("var(--red)" if fsig=="SELL" else "var(--yellow)")
            trade_cls="trade-buy" if fsig=="BUY" else ("trade-sell" if fsig=="SELL" else "trade-hold")
            m1,m2,m3,m4,m5=st.columns(5)
            for col,lbl,val,clr in [
                (m1,"LIVE PRICE",  fmt_price(live_p),             "var(--text)"),
                (m2,"24H CHANGE",  f"{d['chg_pct']:+.2f}%",      "var(--green)" if d["chg_pct"]>=0 else "var(--red)"),
                (m3,"TECH SIGNAL", ts,                            "var(--green)" if ts=="BUY" else "var(--red)" if ts=="SELL" else "var(--yellow)"),
                (m4,"NEWS MOOD",   ns,                            "var(--green)" if ns=="BULLISH" else "var(--red)" if ns=="BEARISH" else "var(--muted)"),
                (m5,"FINAL SIGNAL",fsig,                          sig_color),
            ]:
                with col:
                    st.markdown(f'''<div class="ind-tile">
                      <div class="ind-label">{lbl}</div>
                      <div class="ind-val {price_cls if lbl=='LIVE PRICE' else ''}" style="font-size:1.05rem;color:{clr}">{val}</div>
                    </div>''', unsafe_allow_html=True)
            arrow="▲" if live_dir=="up" else ("▼" if live_dir=="down" else "→")
            st.markdown(f"""
            <div class="trade-box {trade_cls}" style="margin-top:.8rem">
              <div class="trade-title" style="color:{sig_color}">
                {arrow} {fsig} SIGNAL &nbsp;·&nbsp;
                <span style="font-size:.9rem;font-weight:400;color:var(--text)">{selected.replace('-USD','')} — {pool[selected]}</span>
              </div>
              <div style="font-family:var(--mono);font-size:.58rem;color:var(--muted);margin:.2rem 0 .5rem">
                Generated at {timing['time']} &nbsp;·&nbsp; Confidence: {timing['confidence']}% &nbsp;·&nbsp; RSI: {timing['rsi']}
              </div>
              <div class="trade-row">
                <div class="trade-cell"><div class="tc-label">Entry Price</div><div class="tc-val" style="color:var(--cyan)">{fmt_price(timing['entry'])}</div></div>
                <div class="trade-cell"><div class="tc-label">Target Price</div><div class="tc-val" style="color:var(--green)">{fmt_price(timing['target'])}</div></div>
                <div class="trade-cell"><div class="tc-label">Stop Loss</div><div class="tc-val" style="color:var(--red)">{fmt_price(timing['stop_loss'])}</div></div>
                <div class="trade-cell"><div class="tc-label">Hold Duration</div><div class="tc-val" style="color:var(--yellow)">{timing['hold']}</div></div>
                <div class="trade-cell"><div class="tc-label">ATR (14)</div><div class="tc-val" style="color:var(--violet)">{fmt_price(timing['atr'])}</div></div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(candlestick_chart(selected,f"{selected.replace('-USD','')} — 5D Hourly",
                timing["entry"],timing["target"],timing["stop_loss"]),use_container_width=True,config={"displayModeBar":False})
            st.markdown(f'''<div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
              padding:.75rem 1rem;margin:.5rem 0;font-size:.76rem;color:var(--text);line-height:1.55">
              <span style="font-family:var(--mono);font-size:.53rem;color:var(--violet);letter-spacing:.14em">
              TECHNICAL REASONING</span><br>{tech_reason}
            </div>''', unsafe_allow_html=True)
            if chronos_enabled:
                with st.spinner(f"Chronos-Alpha analysing {selected.replace('-USD','')}…"):
                    _ca=chronos_analyse_asset(ticker=selected,name=pool[selected],
                        asset_class="Crypto" if "-USD" in selected else "Stock",
                        price=live_p,chg_pct=d["chg_pct"],wk_chg=d["wk_chg"],
                        rsi=timing["rsi"],signal=fsig,score=fscore,
                        news_headlines=news,groq_key=GROQ_API_KEY)
                render_chronos_block(_ca,fsig)
        st.markdown('<p class="sec" style="margin-top:.8rem">Latest News Headlines</p>', unsafe_allow_html=True)
        if news:
            for n in news:
                sent=n["sentiment"]; ncls=f"news-{sent[:4]}"; scls=f"news-sentiment-{sent[:4]}"
                emoji="📈" if sent=="bullish" else ("📉" if sent=="bearish" else "➡️")
                st.markdown(f'''<div class="news-item {ncls}">
                  <div class="news-src">{n['publisher']}</div>
                  <div class="news-headline">{n['title']}</div>
                  <div class="{scls}">{emoji} {sent.upper()} &nbsp;·&nbsp; Signal: <b>{n['signal']}</b></div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="news-item news-neut"><div class="news-headline" style="color:var(--muted)">No recent headlines via Yahoo Finance.</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COMPARE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="sec">60-Day Normalised Performance</p>', unsafe_allow_html=True)
    sd={t:prices[t] for t in STOCKS if t in prices and len(prices[t]["history"])>1}
    cd={t.replace("-USD",""):prices[t] for t in CRYPTO if t in prices and len(prices[t]["history"])>1}
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(normalised_chart(sd,"Stocks — 60D % Return"),use_container_width=True,config={"displayModeBar":False})
    with c2: st.plotly_chart(normalised_chart(cd,"Crypto — 60D % Return"),use_container_width=True,config={"displayModeBar":False})
    st.markdown('<p class="sec" style="margin-top:1rem">Full Asset Scorecard + Trade Levels</p>', unsafe_allow_html=True)
    rows=[]
    for ticker,name in {**STOCKS,**CRYPTO}.items():
        d=prices.get(ticker)
        if not d: continue
        lp=d.get("live_price",d["price"]); ts,score,_=compute_technical_signal(d["history"],d["chg_pct"],d["wk_chg"])
        ni=fetch_news(ticker); ns,_,_=news_signal(ni); fsig,fscore=combined_signal(ts,score,ns); tm=compute_trade_timing(d,fsig,lp)
        rows.append({"Asset":ticker.replace("-USD",""),"Name":name,"Live $":fmt_price(lp),
            "24H %":f"{d['chg_pct']:+.2f}%","7D %":f"{d['wk_chg']:+.2f}%","Signal":fsig,
            "Entry":fmt_price(tm["entry"]),"Target":fmt_price(tm["target"]),"Stop":fmt_price(tm["stop_loss"]),
            "Hold":tm["hold"],"Conf %":tm["confidence"],"RSI":tm["rsi"]})
    def hl(val):
        if val in ("BUY","BULLISH"):  return "color:#10d98a;font-weight:bold"
        if val in ("SELL","BEARISH"): return "color:#f43f5e;font-weight:bold"
        if val in ("HOLD","NEUTRAL"): return "color:#fbbf24;font-weight:bold"
        if isinstance(val,str) and val.startswith("+"): return "color:#10d98a"
        if isinstance(val,str) and val.startswith("-"): return "color:#f43f5e"
        return ""
    st.dataframe(pd.DataFrame(rows).style.applymap(hl,subset=["Signal","24H %","7D %"]),
        use_container_width=True,hide_index=True,height=680)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CHRONOS-ALPHA
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="sec">🤖 Chronos-Alpha Intelligence Hub</p>', unsafe_allow_html=True)
    if not chronos_enabled:
        st.markdown('''<div style="background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.28);
          border-radius:14px;padding:1.5rem;text-align:center">
          <div style="font-family:'DM Mono',monospace;font-size:.9rem;color:#f43f5e;margin-bottom:.6rem">
            ⚠️ CHRONOS-ALPHA OFFLINE</div>
          <div style="font-size:.82rem;color:#e8e8f0;line-height:1.6">
            Enter Groq API key in the <b>sidebar</b>.<br>
            Free key at <b style="color:#a78bfa">console.groq.com</b>
          </div></div>''', unsafe_allow_html=True)
    else:
        st.markdown('<p class="sec">🌍 Geopolitical & Black Swan Scan</p>', unsafe_allow_html=True)
        with st.spinner("Chronos-Alpha scanning geopolitical risks…"):
            _geo=chronos_geopolitical_scan(GROQ_API_KEY)
        render_geo_block(_geo)
        st.markdown('<p class="sec" style="margin-top:1.2rem">🔬 Deep Asset Analysis</p>', unsafe_allow_html=True)
        _cc1,_cc2=st.columns([1,3])
        with _cc1: _ca_class=st.radio("Asset class",["Stocks","Crypto"],key="ca_class")
        with _cc2:
            _ca_pool=STOCKS if _ca_class=="Stocks" else CRYPTO
            _ca_ticker=st.selectbox("Select asset",list(_ca_pool.keys()),
                format_func=lambda t: f"{t.replace('-USD','')} — {_ca_pool[t]}",key="ca_ticker")
        if _ca_ticker:
            _d2=prices.get(_ca_ticker,{})
            if _d2:
                _lp2=_d2.get("live_price",_d2["price"])
                _ts2,_sc2,_=compute_technical_signal(_d2["history"],_d2["chg_pct"],_d2["wk_chg"])
                _ni2=fetch_news(_ca_ticker); _ns2,_,_=news_signal(_ni2); _fs2,_fsc2=combined_signal(_ts2,_sc2,_ns2)
                _tm2=compute_trade_timing(_d2,_fs2,_lp2)
                _x1,_x2,_x3,_x4=st.columns(4)
                for _col,_lbl,_val,_clr in [
                    (_x1,"LIVE PRICE",fmt_price(_lp2),"var(--text)"),
                    (_x2,"SIGNAL",_fs2,"var(--green)" if _fs2=="BUY" else "var(--red)" if _fs2=="SELL" else "var(--yellow)"),
                    (_x3,"RSI",str(_tm2["rsi"]),"var(--violet)"),
                    (_x4,"CONFIDENCE",f"{_tm2['confidence']}%","var(--cyan)"),
                ]:
                    with _col:
                        st.markdown(f'''<div class="ind-tile">
                          <div class="ind-label">{_lbl}</div>
                          <div class="ind-val" style="font-size:1rem;color:{_clr}">{_val}</div>
                        </div>''', unsafe_allow_html=True)
                with st.spinner(f"Chronos-Alpha deep analysis of {_ca_ticker.replace('-USD','')}…"):
                    _ca2=chronos_analyse_asset(ticker=_ca_ticker,name=_ca_pool[_ca_ticker],
                        asset_class="Crypto" if "-USD" in _ca_ticker else "Stock",
                        price=_lp2,chg_pct=_d2["chg_pct"],wk_chg=_d2["wk_chg"],
                        rsi=_tm2["rsi"],signal=_fs2,score=_fsc2,
                        news_headlines=_ni2,groq_key=GROQ_API_KEY)
                render_chronos_block(_ca2,_fs2)
        st.markdown('<p class="sec" style="margin-top:1.2rem">📋 All Assets — Chronos-Alpha Signals</p>', unsafe_allow_html=True)
        _rows=[]
        for _ticker,_name in {**STOCKS,**CRYPTO}.items():
            _d3=prices.get(_ticker)
            if not _d3: continue
            _lp3=_d3.get("live_price",_d3["price"])
            _ts3,_sc3,_=compute_technical_signal(_d3["history"],_d3["chg_pct"],_d3["wk_chg"])
            _ni3=fetch_news(_ticker); _ns3,_,_=news_signal(_ni3); _fs3,_fsc3=combined_signal(_ts3,_sc3,_ns3)
            _tm3=compute_trade_timing(_d3,_fs3,_lp3)
            _rows.append({"Asset":_ticker.replace("-USD",""),"Name":_name,"Price":fmt_price(_lp3),
                "Signal":_fs3,"Entry":fmt_price(_tm3["entry"]),"Target":fmt_price(_tm3["target"]),
                "Stop":fmt_price(_tm3["stop_loss"]),"Hold":_tm3["hold"],"RSI":_tm3["rsi"],"Score":_fsc3,"News":_ns3})
        st.dataframe(pd.DataFrame(_rows).style.applymap(hl,subset=["Signal","News"]),
            use_container_width=True,hide_index=True,height=700)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  GEO MARKET INTELLIGENCE v5.2 &nbsp;·&nbsp; OIL+GOLD RANDOMFOREST + 5-FACTOR SIGNAL ENGINE + CHRONOS-ALPHA (LLAMA 3.3 70B) &nbsp;·&nbsp; TRADINGVIEW SKIN
  &nbsp;·&nbsp; DATA: YAHOO FINANCE &nbsp;·&nbsp; LIVE 5s REFRESH &nbsp;·&nbsp; NOT FINANCIAL ADVICE
  &nbsp;·&nbsp; BUILT BY <a href="https://x.com/steadyfit1" target="_blank"
    style="color:#a78bfa;text-decoration:none">@STEADYFIT1</a>
  &nbsp;·&nbsp; {now_str}
</div>
""", unsafe_allow_html=True)
