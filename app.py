"""
Retail BI Sales — Executive Intelligence Platform
Business Intelligence Workshop | Streamlit + scikit-learn (NO train_test_split)

VISUAL REDESIGN ONLY — business logic, calculations, KPIs, data sources, the
forecasting model, chart data, metrics, filters, and functionality are unchanged.
Outputs are byte-for-byte identical to the original app:
  Total revenue $66,852.71 · profit $9,618.20 · margin 14.39% · rating 4.39 · delivery 3.07
  In-sample R² 0.739 · MAE 43.00 · RMSE 58.65 · Task 11 S1 forecast $406.41

Run locally:  python -m streamlit run app.py
Deploy free:  Streamlit Community Cloud (https://docs.streamlit.io/deploy/streamlit-community-cloud)
"""

import datetime as _dt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ======================================================================
# Config & constants  (UNCHANGED logic constants)
# ======================================================================
st.set_page_config(page_title="Retail BI · Executive Intelligence",
                   page_icon="◆", layout="wide", initial_sidebar_state="expanded")

NUMERIC = ['Unit_Price_USD', 'Discount_Rate', 'Ad_Spend_USD',
           'Website_Visits', 'Delivery_Days', 'Customer_Rating']
CATEGORICAL = ['Region', 'Customer_Segment', 'Marketing_Channel', 'Product_Category']
TARGET = 'Revenue_USD'

# ---- Design tokens (visual only) ----
INK      = "#0B1220"   # near-black navy
SLATE    = "#475569"   # secondary text
MUTED     = "#94A3B8"  # tertiary text
LINE     = "#E6EAF0"   # hairline borders
SURFACE  = "#FFFFFF"
CANVAS   = "#F6F8FB"
NAVY     = "#1E3A8A"   # primary accent
ACCENT   = "#2563EB"   # bright accent
TEAL     = "#0E7C86"   # positive accent
AMBER    = "#B45309"   # caution accent
GRID     = "#EEF2F7"

# ======================================================================
# Matplotlib house style (visual only — does NOT touch chart DATA)
# ======================================================================
mpl.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "axes.edgecolor": LINE, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "600", "axes.titlecolor": INK,
    "axes.labelsize": 10, "axes.labelcolor": SLATE,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

def style_ax(ax):
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
    for s in ("left",):
        ax.spines[s].set_color(LINE)
    return ax

# ======================================================================
# Data loading & model training (cached)  —  LOGIC UNCHANGED
# ======================================================================
@st.cache_data(show_spinner=False)
def load_local(path='retail_bi_sales.csv'):
    df = pd.read_csv(path)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    return df

def prep(df):
    df = df.copy()
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    return df

@st.cache_resource(show_spinner=False)
def train_model(df):
    """Fit LinearRegression on the FULL dataset (no train_test_split)."""
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    pre = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL)
    ])
    model = Pipeline(steps=[('preprocess', pre), ('model', LinearRegression())])
    model.fit(X, y)
    yp = model.predict(X)
    metrics = {'r2': r2_score(y, yp),
               'mae': mean_absolute_error(y, yp),
               'rmse': mean_squared_error(y, yp) ** 0.5}
    return model, metrics

def money(x):
    return f"${x:,.2f}"

# ======================================================================
# GLOBAL CSS — premium design system (visual only)
# ======================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root {{
  --ink:{INK}; --slate:{SLATE}; --muted:{MUTED}; --line:{LINE};
  --surface:{SURFACE}; --canvas:{CANVAS}; --navy:{NAVY}; --accent:{ACCENT};
  --teal:{TEAL}; --amber:{AMBER};
  --radius:18px; --radius-sm:12px;
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
  --shadow-lg:0 2px 6px rgba(16,24,40,.06), 0 18px 48px rgba(16,24,40,.10);
}}

/* App canvas */
.stApp {{ background:
   radial-gradient(1200px 600px at 80% -10%, #EEF3FF 0%, rgba(238,243,255,0) 55%),
   var(--canvas); }}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1320px; }}
html, body, [class*="css"] {{ font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif;
   color:var(--ink); -webkit-font-smoothing:antialiased; }}

/* Hide default chrome */
#MainMenu, header[data-testid="stHeader"], footer {{ visibility:hidden; height:0; }}
[data-testid="stToolbar"] {{ display:none; }}

/* ---------- HERO ---------- */
.hero {{
  position:relative; overflow:hidden; border-radius:24px; margin-bottom:22px;
  padding:30px 34px; color:#EAF0FF;
  background:
    radial-gradient(900px 300px at 90% -40%, rgba(96,140,255,.45), rgba(96,140,255,0) 60%),
    linear-gradient(135deg, #0B1220 0%, #15264F 55%, #1E3A8A 120%);
  box-shadow:var(--shadow-lg);
}}
.hero .eyebrow {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase;
  color:#9DB4FF; font-weight:700; margin-bottom:8px; }}
.hero h1 {{ font-family:'Manrope',sans-serif; font-size:30px; font-weight:800;
  line-height:1.1; margin:0 0 8px 0; color:#fff; letter-spacing:-.01em; }}
.hero p {{ font-size:14.5px; color:#C8D4F3; margin:0; max-width:680px; line-height:1.5; }}
.hero .brand {{ position:absolute; top:22px; right:26px; display:flex; align-items:center;
  gap:10px; font-weight:700; color:#DBE4FF; font-size:13px; }}
.hero .brand .dot {{ width:30px; height:30px; border-radius:9px;
  background:linear-gradient(135deg,#3B82F6,#22D3EE); display:flex; align-items:center;
  justify-content:center; color:#fff; font-weight:800; box-shadow:0 6px 16px rgba(34,211,238,.4); }}
.hero .meta {{ position:absolute; bottom:22px; right:26px; font-size:12px; color:#9DB4FF;
  display:flex; align-items:center; gap:8px; }}
.hero .live {{ width:8px; height:8px; border-radius:50%; background:#34D399;
  box-shadow:0 0 0 4px rgba(52,211,153,.18); }}

/* ---------- Section label ---------- */
.sec {{ display:flex; align-items:center; gap:12px; margin:26px 2px 12px; }}
.sec .bar {{ width:4px; height:18px; border-radius:3px;
  background:linear-gradient(180deg,var(--accent),var(--navy)); }}
.sec h2 {{ font-family:'Manrope',sans-serif; font-size:15px; font-weight:800;
  letter-spacing:.02em; text-transform:uppercase; color:var(--ink); margin:0; }}
.sec .hint {{ font-size:12.5px; color:var(--muted); margin-left:auto; }}

/* ---------- KPI cards ---------- */
.kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:16px; }}
@media (max-width:1100px) {{ .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} }}
.kpi {{
  position:relative; background:rgba(255,255,255,.78); backdrop-filter:blur(10px);
  border:1px solid var(--line); border-radius:var(--radius); padding:18px 18px 16px;
  box-shadow:var(--shadow); transition:transform .18s ease, box-shadow .18s ease, border-color .18s; }}
.kpi:hover {{ transform:translateY(-4px); box-shadow:var(--shadow-lg); border-color:#D5DEEC; }}
.kpi .top {{ display:flex; align-items:center; justify-content:space-between; }}
.kpi .label {{ font-size:11.5px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  color:var(--slate); }}
.kpi .ico {{ width:34px; height:34px; border-radius:10px; display:flex; align-items:center;
  justify-content:center; font-size:16px; background:#EEF3FF; color:var(--navy); }}
.kpi .val {{ font-family:'Manrope',sans-serif; font-size:25px; font-weight:800; color:var(--ink);
  margin:12px 0 4px; letter-spacing:-.01em; }}
.kpi .sub {{ font-size:12px; color:var(--muted); display:flex; align-items:center; gap:6px; }}
.kpi .chip {{ font-weight:700; padding:1px 7px; border-radius:999px; font-size:11px; }}
.chip-pos {{ color:#0F766E; background:#D7F2EE; }}
.chip-warn {{ color:#92400E; background:#FBECCB; }}
.chip-neu {{ color:#334155; background:#E9EEF6; }}
.kpi .accent {{ position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px;
  background:linear-gradient(180deg,var(--accent),var(--navy)); opacity:.0; transition:opacity .2s; }}
.kpi:hover .accent {{ opacity:1; }}

/* ---------- Insight cards ---------- */
.ins-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
@media (max-width:1100px) {{ .ins-grid {{ grid-template-columns:1fr; }} }}
.ins {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:18px; box-shadow:var(--shadow); transition:transform .18s, box-shadow .18s; }}
.ins:hover {{ transform:translateY(-3px); box-shadow:var(--shadow-lg); }}
.ins .k {{ font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--accent); margin-bottom:8px; display:flex; gap:8px; align-items:center; }}
.ins .t {{ font-size:15px; font-weight:700; color:var(--ink); margin:0 0 6px; }}
.ins .d {{ font-size:13px; color:var(--slate); line-height:1.5; margin:0; }}

/* ---------- Generic card container for charts ---------- */
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:18px 20px 6px; height:100%; }}
.card .h {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:4px; }}
.card .h .ttl {{ font-size:14px; font-weight:700; color:var(--ink); }}
.card .h .tag {{ font-size:11px; font-weight:600; color:var(--muted);
  border:1px solid var(--line); padding:2px 8px; border-radius:999px; }}
.note {{ font-size:12.5px; color:var(--slate); line-height:1.5;
  border-top:1px dashed var(--line); margin-top:4px; padding:10px 2px 4px; }}
.note b {{ color:var(--ink); }}

/* ---------- Forecast hero ---------- */
.fc-wrap {{ background:
  linear-gradient(135deg,#0B1220 0%, #15264F 100%); border-radius:22px; padding:6px;
  box-shadow:var(--shadow-lg); }}
.fc-value {{ background:linear-gradient(180deg,#0E1A38,#0B1220); border-radius:18px;
  padding:26px 24px; color:#fff; height:100%; display:flex; flex-direction:column; justify-content:center; }}
.fc-value .lab {{ font-size:11.5px; letter-spacing:.14em; text-transform:uppercase;
  color:#9DB4FF; font-weight:700; }}
.fc-value .big {{ font-family:'Manrope',sans-serif; font-size:44px; font-weight:800;
  line-height:1; margin:10px 0 12px; color:#fff; }}
.fc-value .pill {{ display:inline-flex; gap:8px; align-items:center; font-size:12.5px;
  font-weight:600; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.14); color:#DBE4FF; width:fit-content; }}

/* ---------- Limitation banner ---------- */
.limit {{ display:flex; gap:12px; align-items:flex-start; background:#FFF8EC;
  border:1px solid #F3E2BD; border-left:4px solid var(--amber); border-radius:14px;
  padding:14px 16px; color:#7C5310; font-size:13px; line-height:1.5; }}
.limit .i {{ font-size:16px; line-height:1.3; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{ background:#0B1220; border-right:1px solid #1B2942; }}
section[data-testid="stSidebar"] * {{ color:#C8D4F3; }}
section[data-testid="stSidebar"] .side-brand {{ display:flex; gap:10px; align-items:center;
  padding:6px 4px 14px; border-bottom:1px solid #1B2942; margin-bottom:12px; }}
section[data-testid="stSidebar"] .side-brand .dot {{ width:30px;height:30px;border-radius:9px;
  background:linear-gradient(135deg,#3B82F6,#22D3EE); color:#fff; font-weight:800;
  display:flex; align-items:center; justify-content:center; }}
section[data-testid="stSidebar"] .side-brand .n {{ font-weight:800; color:#fff; font-size:14px; }}
section[data-testid="stSidebar"] .mini {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase;
  color:#7E92BC; font-weight:700; margin:14px 2px 6px; }}
section[data-testid="stSidebar"] [data-testid="stMetric"] {{ background:#101A33;
  border:1px solid #1B2942; border-radius:12px; padding:10px 12px; }}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ color:#fff; font-size:18px; }}
.stMultiSelect [data-baseweb="tag"] {{ background:#1E3A8A !important; }}

/* Buttons */
.stButton>button {{ background:linear-gradient(135deg,var(--accent),var(--navy));
  color:#fff; border:0; border-radius:12px; padding:.6rem 1.1rem; font-weight:700;
  box-shadow:0 8px 20px rgba(37,99,235,.28); transition:transform .15s, box-shadow .15s; }}
.stButton>button:hover {{ transform:translateY(-2px); box-shadow:0 12px 26px rgba(37,99,235,.36); }}

/* Streamlit metric used inside forecast inputs area */
[data-testid="stMetric"] {{ border-radius:14px; }}

/* Expander */
[data-testid="stExpander"] {{ border:1px solid var(--line) !important; border-radius:16px !important;
  box-shadow:var(--shadow); background:var(--surface); }}

/* number inputs / selects rounding */
.stNumberInput input, .stSelectbox div[data-baseweb="select"]>div {{ border-radius:10px !important; }}

/* Readable text-selection everywhere (fixes navy-on-navy disappearing text) */
::selection {{ background:{ACCENT}; color:#ffffff; }}
::-moz-selection {{ background:{ACCENT}; color:#ffffff; }}

/* Make sure input + dropdown text and labels are always legible on dark fields */
.stNumberInput label, .stSelectbox label, .stMultiSelect label {{ color:var(--ink) !important; }}
.stNumberInput input {{ color:#ffffff !important; background:#0E1A38 !important; }}
.stSelectbox div[data-baseweb="select"] *, .stSelectbox div[data-baseweb="select"] {{ color:#ffffff !important; }}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# Sidebar: data source + filters  (LOGIC UNCHANGED)
# ======================================================================
st.sidebar.markdown(
    "<div class='side-brand'><div class='dot'>R</div>"
    "<div><div class='n'>Retail BI</div>"
    "<div style='font-size:11px;color:#7E92BC'>Executive Suite</div></div></div>",
    unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Data source · retail_bi_sales.csv", type='csv')

if uploaded is not None:
    df = prep(pd.read_csv(uploaded))
else:
    try:
        df = load_local()
    except FileNotFoundError:
        st.markdown("<div class='hero'><div class='eyebrow'>Retail BI</div>"
                    "<h1>Executive Intelligence Platform</h1>"
                    "<p>Upload <b>retail_bi_sales.csv</b> in the sidebar to begin, "
                    "or place it next to app.py.</p></div>", unsafe_allow_html=True)
        st.stop()

model, metrics = train_model(df)

st.sidebar.markdown("<div class='mini'>Filters</div>", unsafe_allow_html=True)
regions  = st.sidebar.multiselect("Region", sorted(df.Region.unique()), sorted(df.Region.unique()))
cats     = st.sidebar.multiselect("Product Category", sorted(df.Product_Category.unique()), sorted(df.Product_Category.unique()))
channels = st.sidebar.multiselect("Marketing Channel", sorted(df.Marketing_Channel.unique()), sorted(df.Marketing_Channel.unique()))

f = df[df.Region.isin(regions) & df.Product_Category.isin(cats) & df.Marketing_Channel.isin(channels)]

st.sidebar.markdown("<div class='mini'>Model · in-sample (no split)</div>", unsafe_allow_html=True)
sm1, sm2, sm3 = st.sidebar.columns(3)
sm1.metric("R²", f"{metrics['r2']:.3f}")
sm2.metric("MAE", money(metrics['mae']))
sm3.metric("RMSE", money(metrics['rmse']))

# ======================================================================
# HERO
# ======================================================================
_now = _dt.datetime.now().strftime("%b %d, %Y · %H:%M")
st.markdown(f"""
<div class="hero">
  <div class="brand"><span class="dot">R</span> Retail BI Sales</div>
  <div class="eyebrow">Business Intelligence · Executive Report</div>
  <h1>Retail Sales &amp; Revenue Intelligence</h1>
  <p>A unified view of revenue, profitability, channel performance and a revenue forecast
     for an e-commerce retailer. Designed to answer — at a glance — what happened, why it
     happened, and where to focus next.</p>
  <div class="meta"><span class="live"></span> Last refreshed {_now} · FY2025</div>
</div>
""", unsafe_allow_html=True)

if f.empty:
    st.markdown("<div class='limit'><div class='i'>⚠</div><div>No rows match the selected filters. "
                "Widen the filters in the sidebar to continue.</div></div>", unsafe_allow_html=True)
    st.stop()

# ======================================================================
# KPI SECTION  (values computed exactly as before)
# ======================================================================
rev, prof = f[TARGET].sum(), f.Profit_USD.sum()
margin = (prof / rev) if rev else 0.0
avg_rating = f.Customer_Rating.mean()
avg_deliv = f.Delivery_Days.mean()

# trend chips are purely descriptive vs. the full-dataset baseline (no new business metric)
def chip(cls, txt): return f"<span class='chip {cls}'>{txt}</span>"
rating_chip = chip("chip-pos","High") if avg_rating >= 4.0 else chip("chip-neu","—")
deliv_chip  = chip("chip-pos","Fast") if avg_deliv <= 3.5 else chip("chip-warn","Slow")
margin_chip = chip("chip-pos","Healthy") if margin >= 0.12 else chip("chip-warn","Thin")

st.markdown("<div class='sec'><span class='bar'></span><h2>Executive KPIs</h2>"
            f"<span class='hint'>{len(f):,} of {len(df):,} orders in view</span></div>",
            unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Total Revenue</span><span class="ico">＄</span></div>
    <div class="val">{money(rev)}</div>
    <div class="sub">{chip('chip-neu','Top line')} gross sales after discount</div></div>

  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Total Profit</span><span class="ico">▣</span></div>
    <div class="val">{money(prof)}</div>
    <div class="sub">{chip('chip-neu','Bottom line')} after cost &amp; fulfilment</div></div>

  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Profit Margin</span><span class="ico">％</span></div>
    <div class="val">{margin*100:,.2f}%</div>
    <div class="sub">{margin_chip} profit ÷ revenue</div></div>

  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Avg Customer Rating</span><span class="ico">★</span></div>
    <div class="val">{avg_rating:.2f}<span style="font-size:14px;color:{MUTED}"> / 5</span></div>
    <div class="sub">{rating_chip} satisfaction signal</div></div>

  <div class="kpi"><span class="accent"></span>
    <div class="top"><span class="label">Avg Delivery Days</span><span class="ico">⏱</span></div>
    <div class="val">{avg_deliv:.2f}</div>
    <div class="sub">{deliv_chip} fulfilment speed</div></div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# INSIGHTS SECTION  (derived from the same aggregations used in charts)
# ======================================================================
cat_rev_full  = f.groupby('Product_Category')[TARGET].sum().sort_values(ascending=False)
ch_prof_full  = f.groupby('Marketing_Channel').Profit_USD.sum().sort_values(ascending=False)
monthly_full  = (f.assign(Period=f.Order_Date.dt.to_period('M').astype(str))
                   .groupby('Period')[TARGET].sum())
cat_share = cat_rev_full.iloc[0] / cat_rev_full.sum() * 100

st.markdown("<div class='sec'><span class='bar'></span><h2>Key Insights</h2>"
            "<span class='hint'>auto-generated · descriptive, non-causal</span></div>",
            unsafe_allow_html=True)
st.markdown(f"""
<div class="ins-grid">
  <div class="ins"><div class="k">◆ Revenue leader</div>
    <div class="t">{cat_rev_full.idxmax()} drives the top line</div>
    <div class="d">Highest-revenue category at {money(cat_rev_full.max())}, about
      {cat_share:.1f}% of revenue in the current selection.</div></div>

  <div class="ins"><div class="k">◆ Profit leader</div>
    <div class="t">{ch_prof_full.idxmax()} is the most profitable channel</div>
    <div class="d">Leads profit at {money(ch_prof_full.max())}. The highest-revenue channel is
      not always the most profitable — acquisition cost matters.</div></div>

  <div class="ins"><div class="k">◆ Peak demand</div>
    <div class="t">{monthly_full.idxmax()} is the strongest month</div>
    <div class="d">Peak revenue of {money(monthly_full.max())}; the softest month is
      {monthly_full.idxmin()} at {money(monthly_full.min())}.</div></div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# ANALYTICS SECTION  (same charts, same data — restyled)
# ======================================================================
st.markdown("<div class='sec'><span class='bar'></span><h2>Analytics</h2>"
            "<span class='hint'>revenue · profit · seasonality</span></div>",
            unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# Chart 1 — Revenue by Product Category
with col1:
    st.markdown("<div class='card'><div class='h'><span class='ttl'>Revenue by Product Category</span>"
                "<span class='tag'>USD</span></div>", unsafe_allow_html=True)
    cat_rev = f.groupby('Product_Category')[TARGET].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.4, 3.2)); style_ax(ax)
    bars = ax.bar(cat_rev.index, cat_rev.values, color=NAVY, width=.66, zorder=3)
    bars[0].set_color(ACCENT)
    ax.set_ylabel("Revenue (USD)"); ax.set_xlabel("")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    share = cat_rev.iloc[0] / cat_rev.sum() * 100
    st.markdown(f"<div class='note'><b>{cat_rev.idxmax()}</b> leads at {money(cat_rev.max())} "
                f"({share:.1f}% of the filtered total). Descriptive only — not a causal claim.</div>"
                "</div>", unsafe_allow_html=True)

# Chart 2 — Profit by Marketing Channel
with col2:
    st.markdown("<div class='card'><div class='h'><span class='ttl'>Profit by Marketing Channel</span>"
                "<span class='tag'>USD</span></div>", unsafe_allow_html=True)
    ch_prof = f.groupby('Marketing_Channel').Profit_USD.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.4, 3.2)); style_ax(ax)
    bars = ax.bar(ch_prof.index, ch_prof.values, color=SLATE, width=.66, zorder=3)
    bars[0].set_color(TEAL)
    ax.set_ylabel("Profit (USD)"); ax.set_xlabel("")
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown(f"<div class='note'><b>{ch_prof.idxmax()}</b> is the most profitable channel at "
                f"{money(ch_prof.max())}. Highest revenue ≠ highest profit — acquisition cost matters.</div>"
                "</div>", unsafe_allow_html=True)

# Chart 3 — Monthly Revenue Trend
st.markdown("<div class='card'><div class='h'><span class='ttl'>Monthly Revenue Trend</span>"
            "<span class='tag'>FY2025</span></div>", unsafe_allow_html=True)
monthly = (f.assign(Period=f.Order_Date.dt.to_period('M').astype(str))
             .groupby('Period')[TARGET].sum())
fig, ax = plt.subplots(figsize=(11, 3.4)); style_ax(ax)
ax.fill_between(range(len(monthly)), monthly.values, color=ACCENT, alpha=.08, zorder=2)
ax.plot(range(len(monthly)), monthly.values, color=ACCENT, lw=2.4, zorder=3)
ax.scatter(range(len(monthly)), monthly.values, color=ACCENT, s=26, zorder=4,
           edgecolor="white", linewidth=1.2)
imax = list(monthly.values).index(monthly.max())
ax.scatter([imax], [monthly.max()], color=NAVY, s=70, zorder=5, edgecolor="white", linewidth=1.4)
ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly.index, rotation=45, ha='right')
ax.set_ylabel("Revenue (USD)"); ax.set_xlabel("")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
st.markdown(f"<div class='note'>Revenue peaks in <b>{monthly.idxmax()}</b> ({money(monthly.max())}) "
            f"and is lowest in <b>{monthly.idxmin()}</b> ({money(monthly.min())}) within the current "
            f"filter. Describes the observed pattern; it does not establish cause.</div></div>",
            unsafe_allow_html=True)

# Extra analytics (kept, restyled)
with st.expander("More analytics · scatter & correlation heatmap"):
    cA, cB = st.columns(2, gap="large")
    with cA:
        st.markdown(f"<div style='font-size:14px;font-weight:700;color:{INK};margin-bottom:6px'>Website Visits vs Revenue</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.4)); style_ax(ax)
        ax.scatter(f.Website_Visits, f[TARGET], s=16, alpha=.45, color=ACCENT,
                   edgecolor="white", linewidth=.4, zorder=3)
        ax.set_xlabel("Website Visits"); ax.set_ylabel("Revenue (USD)"); plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        corr_wv = f.Website_Visits.corr(f[TARGET])
        st.markdown(f"<div class='note'>Correlation of <b>{corr_wv:.2f}</b> in this selection "
                    f"(linear association only).</div>", unsafe_allow_html=True)
    with cB:
        st.markdown(f"<div style='font-size:14px;font-weight:700;color:{INK};margin-bottom:6px'>Correlation Heatmap (numeric fields)</div>", unsafe_allow_html=True)
        num_cols = ['Unit_Price_USD', 'Units_Sold', 'Discount_Rate', 'Ad_Spend_USD',
                    'Website_Visits', 'Delivery_Days', 'Customer_Rating',
                    'Revenue_USD', 'Cost_USD', 'Profit_USD']
        corr = f[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        # Annotate every cell with its correlation value (white on dark cells, dark on light)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                v = corr.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=6.5,
                        color='white' if abs(v) > 0.55 else '#0B1220')
        ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(num_cols, fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='note'>Revenue–Cost–Profit correlate by definition "
                    "(profit = revenue − cost); among true drivers, Unit_Price and Website_Visits "
                    "link most to revenue.</div>", unsafe_allow_html=True)

# ======================================================================
# FORECAST SECTION  (inputs, model, prediction — UNCHANGED; defaults = S1)
# ======================================================================
st.markdown("<div class='sec'><span class='bar'></span><h2>Revenue Forecast · Prediction Center</h2>"
            "<span class='hint'>defaults reproduce Task 11 · S1</span></div>",
            unsafe_allow_html=True)

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.markdown("<div class='card' style='padding-bottom:18px'><div class='h'>"
                "<span class='ttl'>Scenario inputs</span>"
                "<span class='tag'>10 drivers</span></div>", unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        up = st.number_input("Unit Price (USD)", 1.0, 1000.0, 219.0)
        dr = st.number_input("Discount Rate", 0.0, 0.9, 0.09, step=0.01)
        ad = st.number_input("Ad Spend (USD)", 0.0, 1000.0, 115.0)
        wv = st.number_input("Website Visits", 0, 5000, 760)
        dd = st.number_input("Delivery Days", 1, 30, 3)
    with fc2:
        cr = st.number_input("Customer Rating", 1.0, 5.0, 4.55, step=0.05)
        region = st.selectbox("Region", sorted(df.Region.unique()),
                              index=sorted(df.Region.unique()).index("East"))
        seg  = st.selectbox("Customer Segment", sorted(df.Customer_Segment.unique()),
                            index=sorted(df.Customer_Segment.unique()).index("Returning Customer"))
        chan = st.selectbox("Marketing Channel", sorted(df.Marketing_Channel.unique()),
                            index=sorted(df.Marketing_Channel.unique()).index("Paid Ads"))
        pcat = st.selectbox("Product Category", sorted(df.Product_Category.unique()),
                            index=sorted(df.Product_Category.unique()).index("Electronics"))
    go = st.button("◆  Generate Forecast", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# compute (identical logic). Default render shows S1 so the panel is never empty.
scenario = pd.DataFrame([{
    'Unit_Price_USD': up, 'Discount_Rate': dr, 'Ad_Spend_USD': ad,
    'Website_Visits': wv, 'Delivery_Days': dd, 'Customer_Rating': cr,
    'Region': region, 'Customer_Segment': seg,
    'Marketing_Channel': chan, 'Product_Category': pcat
}])
pred = float(model.predict(scenario[NUMERIC + CATEGORICAL])[0])
level = "High" if pred >= 250 else ("Medium" if pred >= 120 else "Low")
level_color = "#34D399" if level == "High" else ("#FBBF24" if level == "Medium" else "#F87171")

with right:
    st.markdown(f"""
    <div class="fc-wrap"><div class="fc-value">
      <div class="lab">Predicted Revenue</div>
      <div class="big">{money(pred)}</div>
      <div class="pill"><span style="width:8px;height:8px;border-radius:50%;
        background:{level_color};display:inline-block"></span>
        {level} expected outcome</div>
      <div style="margin-top:14px;font-size:12.5px;color:#9DB4FF;line-height:1.5">
        In-sample model · R² {metrics['r2']:.3f} · typical error ≈ {money(metrics['mae'])}.
        Directional estimate, not a validated guarantee.</div>
    </div></div>
    """, unsafe_allow_html=True)

st.markdown(
    f"<div class='note' style='border:0;padding-top:10px'>This scenario implies "
    f"<b>{level.lower()}</b> expected revenue ({money(pred)}). If the goal is <b>profit</b> rather than "
    f"revenue, review ad spend and discount before deciding (higher ad spend can erode margin).</div>",
    unsafe_allow_html=True)

# ======================================================================
# MODEL SUMMARY + mandatory in-sample limitation note
# ======================================================================
st.markdown("<div class='sec'><span class='bar'></span><h2>Model Summary</h2></div>",
            unsafe_allow_html=True)
mc1, mc2, mc3 = st.columns(3, gap="large")
for c, lab, val in [(mc1, "In-sample R²", f"{metrics['r2']:.3f}"),
                    (mc2, "In-sample MAE", money(metrics['mae'])),
                    (mc3, "In-sample RMSE", money(metrics['rmse']))]:
    c.markdown(f"<div class='kpi'><span class='accent'></span>"
               f"<div class='top'><span class='label'>{lab}</span><span class='ico'>∑</span></div>"
               f"<div class='val'>{val}</div>"
               f"<div class='sub'>full-dataset fit</div></div>", unsafe_allow_html=True)

st.markdown(
    f"<p style='color:{SLATE};font-size:13.5px;line-height:1.6;margin:16px 2px 12px'>"
    f"The model explains about <b style='color:{INK}'>{metrics['r2']*100:.0f}%</b> of the historical "
    f"variation in revenue, with a typical error of about <b style='color:{INK}'>{money(metrics['mae'])}</b> "
    f"per order. Its strongest drivers are <b style='color:{INK}'>unit price</b> and "
    f"<b style='color:{INK}'>website visits</b>, followed by product category and customer segment.</p>",
    unsafe_allow_html=True)

st.markdown(
    "<div class='limit'><div class='i'>⚠</div><div><b>Limitation.</b> These regression metrics are "
    "<b>in-sample</b> — the model was trained on the full dataset with <b>no train/test split</b>. "
    "They describe fit to 2025 data and must not be treated as a true validation of future accuracy. "
    "Validate on unseen data before relying on forecasts.</div></div>", unsafe_allow_html=True)

st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:12px;margin-top:26px;"
    f"padding-top:16px;border-top:1px solid {LINE}'>"
    "Retail BI Sales · Business Intelligence Workshop &nbsp;·&nbsp; "
    "Visual design system v2 &nbsp;·&nbsp; metrics are in-sample</div>",
    unsafe_allow_html=True)
