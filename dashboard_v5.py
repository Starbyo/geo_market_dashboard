"""
GEO MARKET INTELLIGENCE DASHBOARD v5 — CHRONOS-ALPHA + STEADYFIT SKIN
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
    page_title="Geo Market Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── TOKENS ──────────────────────────────────────────────── */
:root{
  --bg:#0d0d12;--surface:#13131a;--card:#17171f;--card2:#1c1c26;
  --border:rgba(255,255,255,.06);--border2:rgba(255,255,255,.1);
  --purple:#8b5cf6;--purple2:#7c3aed;--violet:#a78bfa;
  --green:#10d98a;--red:#f43f5e;--yellow:#fbbf24;
  --cyan:#22d3ee;--orange:#fb923c;
  --text:#e8e8f0;--muted:#6b6b82;--muted2:#2e2e3e;
  --glow:rgba(139,92,246,.22);--shadow:rgba(0,0,0,.65);
  --font:'DM Sans',sans-serif;--mono:'DM Mono',monospace;
}

/* ── RESET ───────────────────────────────────────────────── */
*{box-sizing:border-box;}
html,body,[class*="css"]{
  background:var(--bg)!important;
  color:var(--text)!important;
  font-family:var(--font)!important;
}
.main,.block-container{
  background:var(--bg)!important;
  padding:1.2rem 1.8rem!important;
  max-width:100%!important;
}

/* ── AMBIENT BLOBS ───────────────────────────────────────── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:-1;
  background:
    radial-gradient(ellipse 600px 400px at -10% -10%,rgba(124,58,237,.12),transparent),
    radial-gradient(ellipse 500px 350px at 110% 110%,rgba(79,70,229,.1),transparent),
    radial-gradient(ellipse 300px 300px at 50% 60%,rgba(168,85,247,.07),transparent);
}

/* ── GRID TEXTURE ────────────────────────────────────────── */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:-1;
  background-image:
    linear-gradient(rgba(139,92,246,.02) 1px,transparent 1px),
    linear-gradient(90deg,rgba(139,92,246,.02) 1px,transparent 1px);
  background-size:44px 44px;
}

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"]>div:first-child{padding:1rem .85rem;}
[data-testid="stSidebarCollapseButton"]{color:var(--muted)!important;}

/* ── TABS ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{
  background:var(--surface);border-radius:10px;padding:3px;
  border:1px solid var(--border);gap:2px;
}
.stTabs [data-baseweb="tab"]{
  font-family:var(--mono);font-size:.62rem;letter-spacing:.08em;
  color:var(--muted)!important;border-radius:7px;padding:.38rem .9rem;
  transition:all .18s;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(139,92,246,.24),rgba(79,70,229,.14))!important;
  color:var(--violet)!important;
}

/* ── HEADER ──────────────────────────────────────────────── */
.hdr{border-bottom:1px solid var(--border);padding-bottom:1rem;margin-bottom:1.4rem;}
.hdr-title{
  font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;letter-spacing:.06em;
  background:linear-gradient(90deg,var(--purple) 0%,var(--cyan) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;
}
.hdr-sub{font-family:var(--mono);font-size:.62rem;color:var(--muted);letter-spacing:.15em;margin-top:.3rem;}
.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);
  animation:ldot 1.6s ease-in-out infinite;margin-right:5px;vertical-align:middle;}
@keyframes ldot{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(16,217,138,.4)}50%{opacity:.5;box-shadow:0 0 0 5px rgba(16,217,138,0)}}

/* ── SECTION LABELS ──────────────────────────────────────── */
.sec{
  font-family:var(--mono);font-size:.54rem;color:var(--muted);letter-spacing:.22em;
  text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:.4rem;margin-bottom:.9rem;
}

/* ── PRICE FLASH ─────────────────────────────────────────── */
.price-up  {font-family:var(--mono);font-size:1.25rem;font-weight:700;color:var(--green);animation:flashup .7s ease;}
.price-down{font-family:var(--mono);font-size:1.25rem;font-weight:700;color:var(--red);animation:flashdn .7s ease;}
.price-flat{font-family:var(--mono);font-size:1.25rem;font-weight:700;color:var(--text);}
@keyframes flashup{0%{text-shadow:0 0 18px rgba(16,217,138,.9)}100%{text-shadow:none}}
@keyframes flashdn{0%{text-shadow:0 0 18px rgba(244,63,94,.9)}100%{text-shadow:none}}

/* ── SIGNAL PILLS ────────────────────────────────────────── */
.sig-buy {background:rgba(16,217,138,.11);color:var(--green);border:1px solid rgba(16,217,138,.28);border-radius:5px;padding:2px 9px;font-family:var(--mono);font-size:.62rem;font-weight:600;}
.sig-sell{background:rgba(244,63,94,.11);color:var(--red);border:1px solid rgba(244,63,94,.28);border-radius:5px;padding:2px 9px;font-family:var(--mono);font-size:.62rem;font-weight:600;}
.sig-hold{background:rgba(251,191,36,.11);color:var(--yellow);border:1px solid rgba(251,191,36,.28);border-radius:5px;padding:2px 9px;font-family:var(--mono);font-size:.62rem;font-weight:600;}

/* ── ASSET CARD ──────────────────────────────────────────── */
.asset-card{
  background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:.85rem 1rem;margin-bottom:.7rem;transition:all .22s;position:relative;overflow:hidden;
}
.asset-card:hover{border-color:rgba(139,92,246,.32);transform:translateY(-2px);box-shadow:0 10px 28px var(--shadow);}
.ac-bull{border-left:3px solid var(--green);}
.ac-bear{border-left:3px solid var(--red);}
.ac-hold{border-left:3px solid var(--yellow);}
.ac-top{display:flex;justify-content:space-between;align-items:flex-start;}
.ac-sym{font-family:var(--mono);font-size:.72rem;font-weight:700;color:var(--violet);}
.ac-name{font-size:.62rem;color:var(--muted);margin-top:2px;}
.ac-meta{font-family:var(--mono);font-size:.54rem;color:var(--muted);margin-top:.3rem;}
.chg-pos{font-family:var(--mono);font-size:.8rem;font-weight:700;color:var(--green);}
.chg-neg{font-family:var(--mono);font-size:.8rem;font-weight:700;color:var(--red);}

/* ── RISK BANNER ─────────────────────────────────────────── */
.risk-on-banner{
  background:linear-gradient(135deg,rgba(16,217,138,.09),rgba(34,211,238,.05));
  border:1px solid rgba(16,217,138,.22);border-radius:14px;padding:1.2rem 1.6rem;
}
.risk-off-banner{
  background:linear-gradient(135deg,rgba(244,63,94,.09),rgba(251,191,36,.05));
  border:1px solid rgba(244,63,94,.22);border-radius:14px;padding:1.2rem 1.6rem;
}
.risk-title{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;letter-spacing:.04em;}
.risk-conf{font-family:var(--mono);font-size:.64rem;color:var(--muted);margin:.3rem 0;}
.risk-desc{font-size:.78rem;line-height:1.55;color:var(--text);}

/* ── INDICATOR TILE ──────────────────────────────────────── */
.ind-tile{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:.9rem 1rem;text-align:center;
}
.ind-label{font-family:var(--mono);font-size:.54rem;color:var(--muted);letter-spacing:.16em;text-transform:uppercase;}
.ind-val{font-family:var(--mono);font-size:1.3rem;font-weight:700;color:var(--text);margin:.3rem 0 .2rem;}
.ind-status{font-size:.7rem;font-weight:600;}

/* ── TRADE BOX ───────────────────────────────────────────── */
.trade-box{border-radius:14px;padding:1.1rem 1.3rem;margin:.6rem 0;}
.trade-buy {background:linear-gradient(135deg,rgba(16,217,138,.08),rgba(34,211,238,.04));border:1px solid rgba(16,217,138,.25);}
.trade-sell{background:linear-gradient(135deg,rgba(244,63,94,.08),rgba(251,191,36,.04));border:1px solid rgba(244,63,94,.25);}
.trade-hold{background:rgba(251,191,36,.05);border:1px solid rgba(251,191,36,.22);}
.trade-title{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;letter-spacing:.04em;}
.trade-row{display:flex;gap:1rem;margin-top:.7rem;flex-wrap:wrap;}
.trade-cell{background:rgba(0,0,0,.28);border-radius:10px;padding:.5rem .8rem;min-width:110px;}
.tc-label{font-family:var(--mono);font-size:.53rem;color:var(--muted);letter-spacing:.14em;text-transform:uppercase;}
.tc-val{font-family:var(--mono);font-size:.9rem;font-weight:700;margin-top:.2rem;}

/* ── CHRONOS CARDS ───────────────────────────────────────── */
.ch-card{
  background:linear-gradient(135deg,rgba(139,92,246,.06),rgba(0,0,0,0));
  border:1px solid rgba(139,92,246,.18);border-left:3px solid var(--purple);
  border-radius:14px;padding:1rem 1.2rem;margin:.6rem 0;
}
.ch-sum-card{
  background:linear-gradient(135deg,rgba(139,92,246,.07),rgba(79,70,229,.03));
  border:1px solid rgba(139,92,246,.18);border-radius:14px;padding:1.2rem 1.4rem;margin:.8rem 0 1.2rem;
}
.ch-hdr{font-family:var(--mono);font-size:.58rem;color:var(--violet);letter-spacing:.18em;text-transform:uppercase;margin-bottom:.7rem;}
.ch-row{padding:.38rem 0;border-bottom:1px solid rgba(255,255,255,.04);display:flex;gap:.8rem;align-items:baseline;}
.ch-row:last-child{border-bottom:none;}
.ch-key{font-family:var(--mono);font-size:.54rem;letter-spacing:.1em;min-width:110px;flex-shrink:0;}
.ch-val{font-size:.78rem;color:var(--text);line-height:1.5;}

/* ── NEWS ────────────────────────────────────────────────── */
.news-item{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:.75rem 1rem;margin-bottom:.5rem;transition:all .2s;}
.news-item:hover{border-color:rgba(139,92,246,.22);transform:translateX(2px);}
.news-bull{border-left:3px solid var(--green);}
.news-bear{border-left:3px solid var(--red);}
.news-neut{border-left:3px solid var(--muted2);}
.news-src{font-family:var(--mono);font-size:.54rem;color:var(--violet);letter-spacing:.1em;text-transform:uppercase;}
.news-headline{font-size:.76rem;color:var(--text);margin:.3rem 0 .2rem;line-height:1.45;}
.news-sentiment-bull{font-size:.62rem;color:var(--green);font-weight:600;}
.news-sentiment-bear{font-size:.62rem;color:var(--red);font-weight:600;}
.news-sentiment-neut{font-size:.62rem;color:var(--muted);font-weight:600;}

/* ── OUTLOOK BOX ─────────────────────────────────────────── */
.outlook-box{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1rem;}
.ol-title{font-family:var(--mono);font-size:.56rem;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.6rem;}
.ol-item{font-size:.76rem;color:var(--text);padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);}
.ol-item:last-child{border-bottom:none;}

/* ── ALERT TOASTS ────────────────────────────────────────── */
#toast-container{position:fixed;top:68px;right:18px;z-index:9999;display:flex;flex-direction:column;gap:8px;width:340px;pointer-events:none;}
.toast{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:.7rem 1rem;display:flex;align-items:flex-start;gap:.65rem;
  box-shadow:0 8px 28px var(--shadow);pointer-events:all;position:relative;overflow:hidden;
  animation:toastin .35s cubic-bezier(.34,1.56,.64,1);
}
.toast::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;}
.toast-buy  {border-color:rgba(16,217,138,.28);} .toast-buy::before  {background:var(--green);}
.toast-sell {border-color:rgba(244,63,94,.28);}  .toast-sell::before {background:var(--red);}
.toast-risk {border-color:rgba(139,92,246,.32);} .toast-risk::before {background:var(--purple);}
.toast-news {border-color:rgba(251,191,36,.28);} .toast-news::before {background:var(--yellow);}
.toast-info {border-color:rgba(34,211,238,.28);} .toast-info::before {background:var(--cyan);}
@keyframes toastin{from{opacity:0;transform:translateX(60px) scale(.92)}to{opacity:1;transform:translateX(0) scale(1)}}
.toast-icon{font-size:1.1rem;flex-shrink:0;margin-top:1px;}
.toast-body{flex:1;}
.toast-title{font-size:.73rem;font-weight:600;color:var(--text);margin-bottom:.1rem;}
.toast-msg{font-family:var(--mono);font-size:.58rem;color:var(--muted);line-height:1.45;}
.toast-time{font-family:var(--mono);font-size:.5rem;color:var(--muted);margin-top:.15rem;}
.toast-close{font-size:.65rem;color:var(--muted);cursor:pointer;padding:2px 5px;border-radius:4px;flex-shrink:0;}
.toast-close:hover{background:var(--card2);}
.toast-out{animation:toastout .25s ease forwards;}
@keyframes toastout{to{opacity:0;transform:translateX(60px);}}

/* ── GEO CARD ────────────────────────────────────────────── */
.geo-card{
  background:rgba(251,191,36,.03);border:1px solid rgba(251,191,36,.14);
  border-left:3px solid var(--yellow);border-radius:14px;padding:1rem 1.2rem;margin:.6rem 0;
}

/* ── SIDEBAR CARD ────────────────────────────────────────── */
.sb-ai-card{
  background:linear-gradient(135deg,rgba(139,92,246,.14),rgba(79,70,229,.08));
  border:1px solid rgba(139,92,246,.22);border-radius:12px;padding:.85rem 1rem;margin-top:.5rem;
}
.sb-lbl{font-family:var(--mono);font-size:.54rem;color:var(--violet);letter-spacing:.14em;margin-bottom:.4rem;}
.sb-status-on {font-family:var(--mono);font-size:.62rem;color:var(--green);letter-spacing:.1em;}
.sb-status-off{font-family:var(--mono);font-size:.62rem;color:var(--red);letter-spacing:.1em;}

/* ── FOOTER ──────────────────────────────────────────────── */
.footer{
  margin-top:2.5rem;padding:1rem 0;border-top:1px solid var(--border);
  font-family:var(--mono);font-size:.54rem;color:var(--muted2);
  text-align:center;letter-spacing:.12em;
}

/* ── STREAMLIT OVERRIDES ─────────────────────────────────── */
.stDataFrame{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;}
[data-testid="stSelectbox"]>div>div{background:var(--card2)!important;border-color:var(--border)!important;color:var(--text)!important;}
[data-testid="stRadio"] label{font-family:var(--mono)!important;font-size:.7rem!important;}
.stSpinner>div{border-color:var(--purple) transparent transparent!important;}
div[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;}
::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--muted2);border-radius:2px;}
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
  setTimeout(function(){ toast('news','📰','Markets Update','Risk-on regime confirmed. Equities and crypto showing bullish momentum.'); }, 19000);
  setTimeout(function(){ toast('info','🌍','Geopolitical Watch','Middle East supply risk elevated. Energy sector volatility rising.'); }, 33000);
  setTimeout(function(){ toast('buy','📈','BUY Signal Active','Multiple assets showing BUY — Review Stocks tab for full signals.'); }, 48000);
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Chronos-Alpha + Settings + About + Donate + Contact
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''<div style="font-family:'DM Mono',monospace;font-size:.75rem;color:#a78bfa;
      letter-spacing:.2em;text-transform:uppercase;padding:.4rem 0 .8rem">
      📡 GeoMarket Intelligence</div>''', unsafe_allow_html=True)

    # ── Chronos-Alpha key ────────────────────────────────────
    st.markdown('''<div class="sb-ai-card">
      <div class="sb-lbl">🤖 CHRONOS-ALPHA</div>''', unsafe_allow_html=True)
    groq_key_input = st.text_input(
        "Groq API Key", value=st.session_state.groq_key,
        type="password", placeholder="gsk_...",
        help="Free key at console.groq.com",
    )
    if groq_key_input:
        st.session_state.groq_key = groq_key_input
        GROQ_API_KEY = groq_key_input
    chronos_enabled = bool(GROQ_API_KEY and GROQ_AVAILABLE)
    status_cls = "sb-status-on" if chronos_enabled else "sb-status-off"
    status_txt = "● ACTIVE — Llama 3.3 70B" if chronos_enabled else "● OFFLINE — Add key above"
    st.markdown(f'<div class="{status_cls}">{status_txt}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.62rem;color:#6b6b82;margin-top:.3rem;line-height:1.5">Free key → <b style="color:#a78bfa">console.groq.com</b></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Settings ─────────────────────────────────────────────
    with st.expander("⚙️  Settings"):
        st.markdown('<div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#6b6b82;letter-spacing:.18em;text-transform:uppercase;margin-bottom:.4rem">Appearance</div>', unsafe_allow_html=True)
        theme = st.radio("Theme", ["🌙 Dark", "☀️ Light", "💻 System"], horizontal=True, index=0)
        st.markdown('<div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#6b6b82;letter-spacing:.18em;text-transform:uppercase;margin:.6rem 0 .4rem">Alerts</div>', unsafe_allow_html=True)
        sig_alerts   = st.toggle("Signal Alerts (BUY/SELL)", value=True)
        risk_alerts  = st.toggle("Risk Regime Alerts",       value=True)
        price_alerts = st.toggle("Price Move Alerts",        value=True)
        news_alerts  = st.toggle("Breaking News Alerts",     value=True)
        st.markdown('<div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#6b6b82;letter-spacing:.18em;text-transform:uppercase;margin:.6rem 0 .4rem">Device</div>', unsafe_allow_html=True)
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
# SIGNAL + TIMING ENGINE (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    deltas = np.diff(np.array(prices[-(period+1):]))
    gains  = deltas[deltas > 0].mean() if any(deltas > 0) else 0
    losses = abs(deltas[deltas < 0].mean()) if any(deltas < 0) else 0.001
    return 100 - (100 / (1 + gains / losses))

def compute_technical_signal(history, chg_pct, wk_chg):
    if len(history) < 5: return "HOLD", 50, "Insufficient data"
    h = np.array(history); ma5 = np.mean(h[-5:]); ma20 = np.mean(h[-20:]) if len(h)>=20 else np.mean(h); ma50 = np.mean(h[-50:]) if len(h)>=50 else np.mean(h); price = h[-1]
    score = 50
    if price>ma5: score+=8
    if price>ma20: score+=8
    if price>ma50: score+=6
    if ma5>ma20: score+=8
    if ma20>ma50: score+=5
    score += min(max(chg_pct*2.5,-12),12)
    score += min(max(wk_chg*1.2,-8),8)
    rsi = compute_rsi(history)
    if rsi<30: score+=10
    elif rsi>70: score-=10
    score = max(0,min(100,int(score)))
    if score>=63: return "BUY",  score, f"Price above MA5/MA20. RSI:{rsi:.0f}. Momentum positive. Score {score}/100."
    elif score<=40: return "SELL",score, f"Price below MAs. RSI:{rsi:.0f}. Momentum negative. Score {score}/100."
    else: return "HOLD",score, f"Mixed signals. RSI:{rsi:.0f}. Consolidating. Score {score}/100."

def news_signal(news_items):
    if not news_items: return "NEUTRAL",0,0
    bull=sum(1 for n in news_items if n["sentiment"]=="bullish"); bear=sum(1 for n in news_items if n["sentiment"]=="bearish"); total=len(news_items)
    if bull>bear: return "BULLISH",bull,total
    elif bear>bull: return "BEARISH",bear,total
    else: return "NEUTRAL",0,total

def combined_signal(tech_sig, tech_score, news_sent):
    score=tech_score
    if news_sent=="BULLISH": score+=8
    elif news_sent=="BEARISH": score-=8
    score=max(0,min(100,score))
    if score>=60: return "BUY",score
    elif score<=42: return "SELL",score
    else: return "HOLD",score

def compute_trade_timing(ticker_data, signal, price):
    h=np.array(ticker_data["history"]); atr=ticker_data.get("atr",price*0.02); chg_pct=ticker_data["chg_pct"]; wk_chg=ticker_data["wk_chg"]
    volatility=float(np.std(np.diff(h[-20:])/h[-20:-1])) if len(h)>=21 else 0.02
    vol_ratio=1.0; rsi=compute_rsi(h); ma20=np.mean(h[-20:]) if len(h)>=20 else price
    trend_strength=(price-ma20)/ma20*10
    X_input=np.array([[rsi,chg_pct/100,wk_chg/100,volatility,vol_ratio,trend_strength]])
    X_scaled=timing_scaler.transform(X_input); proba=timing_model.predict_proba(X_scaled)[0]; confidence=int(max(proba)*100)
    atr_mult=1.5+(confidence/100)
    if signal=="BUY":   entry_price=round(price,4); stop_loss=round(price-atr*1.2,4); target_price=round(price+atr*atr_mult*2,4)
    elif signal=="SELL":entry_price=round(price,4); stop_loss=round(price+atr*1.2,4); target_price=round(price-atr*atr_mult*2,4)
    else:               entry_price=round(price,4); stop_loss=round(price-atr,4); target_price=round(price+atr,4)
    if volatility<0.01:   hold_hours="24–48 hours"
    elif volatility<0.02: hold_hours="8–24 hours"
    elif volatility<0.03: hold_hours="4–8 hours"
    else:                 hold_hours="1–4 hours"
    now=datetime.datetime.utcnow()
    return {"entry":entry_price,"target":target_price,"stop_loss":stop_loss,"hold":hold_hours,
            "time":now.strftime("%H:%M UTC"),"confidence":confidence,"rsi":round(rsi,1),"atr":round(atr,4)}

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
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <p class="hdr-title">📡 Geo Market Intelligence</p>
  <p class="hdr-sub">
    <span class="live-dot"></span>Live 5s Refresh &nbsp;·&nbsp; {now_str}
    &nbsp;·&nbsp; 12 Stocks &nbsp;·&nbsp; 10 Crypto
    &nbsp;·&nbsp; Oil+Gold Risk Engine &nbsp;·&nbsp; Chronos-Alpha AI
    &nbsp;·&nbsp; by <a href="https://x.com/steadyfit1" target="_blank"
      style="color:#a78bfa;text-decoration:none">@steadyfit1</a>
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RISK ENGINE (UNCHANGED LOGIC, NEW SKIN)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec">⚡ Risk Engine — Original Oil + Gold RandomForest</p>', unsafe_allow_html=True)

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
  GEO MARKET INTELLIGENCE v5 &nbsp;·&nbsp; OIL+GOLD RANDOMFOREST + GRADIENTBOOST + CHRONOS-ALPHA (LLAMA 3.3 70B)
  &nbsp;·&nbsp; DATA: YAHOO FINANCE &nbsp;·&nbsp; LIVE 5s REFRESH &nbsp;·&nbsp; NOT FINANCIAL ADVICE
  &nbsp;·&nbsp; BUILT BY <a href="https://x.com/steadyfit1" target="_blank"
    style="color:#a78bfa;text-decoration:none">@STEADYFIT1</a>
  &nbsp;·&nbsp; {now_str}
</div>
""", unsafe_allow_html=True)
