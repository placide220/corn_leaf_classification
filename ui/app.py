"""
Ibigori: maize leaf diagnosis (Streamlit dashboard)
=====================================================

This UI talks to the FastAPI backend (api/main.py) over HTTP. Four pages, in the top bar:

    Diagnose a leaf   upload ONE leaf photo and get the diagnosis, shown as a herbarium specimen card
    Field data        the 3 dataset features (from /eda-stats) + model performance (from /model-info)
    Teach the model   bulk-upload labelled images, retrain, database records, version history
    System health     live status of the API: uptime, model version, latency, traffic

Design: a botanist's field guide. Colours come from a maize field (leaf ink, husk, silk gold),
headings use the book serif Literata, everything else the sans-serif Figtree. The theme colours
also live in .streamlit/config.toml so Streamlit's own widgets match.

Run it (with the API already running on port 8000):
    streamlit run ui/app.py
Point it at a deployed API:
    API_URL=https://your-api.example.com streamlit run ui/app.py
(or change the address on the System health page while the app is running).
"""

from __future__ import annotations

import base64
import html
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------------
# 1. Identity, palette and constants
# --------------------------------------------------------------------------------------
APP_NAME = "Ibigori"                       # "maize" in Kinyarwanda
APP_TAGLINE = "Maize leaf diagnosis"
HERE = Path(__file__).resolve().parent

st.set_page_config(page_title=f"{APP_NAME}: {APP_TAGLINE}", page_icon=str(HERE / "assets" / "logo.svg"),
                   layout="wide")

# Palette (same values as .streamlit/config.toml)
INK = "#14382B"          # leaf ink: headings, primary buttons, important numbers
HUSK = "#EDF1E8"         # page background
HUSK_DARK = "#E1E8DA"    # panels and table headers
PAPER = "#FBFCF8"        # the specimen card and other "printed" surfaces
SILK = "#E3B23C"         # silk gold: the one warm accent, used sparingly (highlights, uncertainty)
FIELD_LINE = "#CBD8C3"   # borders and chart grid lines
TEXT = "#1F2B22"
MUTED = "#55635A"        # secondary text
LESION = "#A33B2B"       # errors and "not a maize leaf" only

# Default API address. Can be changed on the System health page.
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Class order used everywhere (must match the model's output order in model_metadata.json)
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# One fixed colour per class, used in EVERY chart so a class always looks the same.
# These are the first four slots of a colour-blind-safe categorical palette (blue, orange,
# aqua, yellow). Colours carry identity only: every chart also shows the class name as text.
CLASS_COLORS = {
    "Blight": "#2a78d6",          # blue
    "Common_Rust": "#eb6834",     # orange
    "Gray_Leaf_Spot": "#1baf7a",  # aqua
    "Healthy": "#eda100",         # yellow
}
# Single-hue scale (light husk -> leaf ink) for "how much" charts such as the confusion matrix.
SEQUENTIAL_GREENS = [[0.0, "#F3F7EF"], [0.5, "#7FA98C"], [1.0, INK]]

# Short, farmer-friendly description of each class, shown next to a prediction.
# General guidance only; the app always recommends confirming with an extension officer.
CLASS_INFO = {
    "Blight": {
        "label": "Northern Leaf Blight",
        "what": "Long, cigar-shaped grey-green to tan lesions that can join up and kill large parts of the leaf.",
        "action": "Remove and destroy badly infected leaves, rotate maize with legumes, bury or remove crop "
                  "residue after harvest, and use resistant varieties next season.",
    },
    "Common_Rust": {
        "label": "Common Rust",
        "what": "Small, raised, reddish-brown powdery pustules scattered on both sides of the leaf.",
        "action": "Monitor how fast it spreads; plant resistant hybrids next season. If it spreads quickly "
                  "before tasselling, ask an extension officer about a suitable fungicide.",
    },
    "Gray_Leaf_Spot": {
        "label": "Gray Leaf Spot",
        "what": "Narrow, rectangular grey or tan lesions running between the leaf veins; worst in warm, humid weather.",
        "action": "Rotate crops, manage residue, avoid planting maize on maize, and choose tolerant varieties.",
    },
    "Healthy": {
        "label": "Healthy leaf",
        "what": "No visible signs of the three diseases the model knows.",
        "action": "Keep scouting the field weekly, especially after rain or long humid spells.",
    },
}



# --------------------------------------------------------------------------------------
# 1b. Typography and styling (injected once per page load)
# --------------------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Literata:opsz,wght@7..72,400;7..72,600;7..72,700&display=swap');

/* Body text: Figtree. Icons keep their own icon font because they set font-family themselves. */
[data-testid="stAppViewContainer"], [data-testid="stHeader"], .stMarkdown, .stCaption,
button, input, textarea, label, [data-testid="stMetric"] {{
    font-family: 'Figtree', system-ui, sans-serif;
}}
/* Headings: Literata, the "printed field guide" voice */
h1, h2, h3, [data-testid="stHeading"] * {{
    font-family: 'Literata', Georgia, serif !important;
    color: {INK};
    letter-spacing: -0.01em;
}}
h1 {{ font-weight: 700 !important; font-size: 2.6rem !important; line-height: 1.15 !important; }}
h2 {{ font-weight: 600 !important; font-size: 1.6rem !important; margin-top: 1.6rem !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.2rem !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
p, li {{ line-height: 1.6; }}

/* Page header: title on the left, live model status on the right */
.ib-intro {{ color: {MUTED}; font-size: 1.05rem; max-width: 62ch; margin: -0.4rem 0 1.4rem 0; }}
.ib-status {{ text-align: right; font-size: 0.9rem; color: {MUTED}; padding-top: 0.9rem; }}
.ib-status b {{ color: {INK}; }}
.ib-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;
           vertical-align: 1px; }}

.ib-insight {{ background: {HUSK_DARK}; border-left: 3px solid {INK}; padding: 14px 16px; border-radius: 0 4px 4px 0;
               color: {TEXT}; font-size: 0.95rem; line-height: 1.6; }}
.ib-insight b {{ color: {INK}; }}

/* Metrics: serif numbers in leaf ink, quiet labels */
[data-testid="stMetricValue"] {{ font-family: 'Literata', Georgia, serif !important; color: {INK}; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.9rem; }}

/* Primary buttons in leaf ink, text in husk */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: {INK}; border-color: {INK}; color: {HUSK}; font-weight: 600;
}}
.stButton > button[kind="primary"]:hover {{ background: #0E2A20; border-color: #0E2A20; color: #fff; }}
:focus-visible {{ outline: 3px solid {SILK} !important; outline-offset: 2px; }}

/* ---------- The herbarium specimen card (the diagnosis result) ---------- */
.specimen {{
    display: grid; grid-template-columns: minmax(220px, 1fr) minmax(280px, 1.15fr); gap: 0;
    background: {PAPER}; border: 1px solid {FIELD_LINE}; border-radius: 4px;
    box-shadow: 0 1px 0 {FIELD_LINE}, 0 12px 28px -18px rgba(20, 56, 43, 0.45);
    margin: 0.4rem 0 1.6rem 0; overflow: hidden;
}}
.specimen-photo {{ background: #F4F6EF; padding: 22px; display: flex; align-items: center; justify-content: center;
                  border-right: 1px solid {FIELD_LINE}; }}
.specimen-photo img {{ max-width: 100%; max-height: 360px; border-radius: 2px;
                      box-shadow: 0 2px 6px rgba(31, 43, 34, 0.18); }}
.specimen-label {{ padding: 26px 28px 22px 28px; font-family: 'Literata', Georgia, serif; color: {TEXT}; }}
.label-frame {{ border: 1px solid {INK}; outline: 1px solid {INK}; outline-offset: 3px; padding: 16px 18px;
               margin: 4px; }}
.label-title {{ font-size: 0.95rem; color: {MUTED}; margin-bottom: 2px; }}
.label-diagnosis {{ font-size: 1.9rem; font-weight: 700; color: {INK}; line-height: 1.2; }}
.label-latin {{ font-style: italic; color: {MUTED}; font-size: 0.98rem; margin-bottom: 12px; }}
.label-rows, .label-rows tr, .label-rows td {{ border: none !important; background: transparent !important; }}
.label-rows {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; margin: 0 !important; }}
.label-rows td {{ padding: 6px 0 !important; border-top: 1px dotted {INK}55 !important; vertical-align: baseline; }}
.label-rows td:first-child {{ color: {MUTED}; width: 38%; font-family: 'Figtree', sans-serif; font-size: 0.88rem; }}
.label-confidence {{ font-size: 1.35rem; font-weight: 600; color: {INK}; }}
.label-note {{ margin-top: 12px; padding: 9px 12px; background: #FBF3DC; border-left: 3px solid {SILK};
              font-family: 'Figtree', sans-serif; font-size: 0.9rem; line-height: 1.45; }}
.label-reject {{ font-size: 1.6rem; font-weight: 700; color: {LESION}; line-height: 1.2; }}
.probs {{ margin-top: 16px; font-family: 'Figtree', sans-serif; font-size: 0.88rem; }}
.prob-row {{ display: grid; grid-template-columns: 9.5rem 1fr 3.4rem; align-items: center; gap: 10px; margin: 5px 0; }}
.prob-track {{ background: #E8EEE3; height: 10px; border-radius: 0 4px 4px 0; }}
.prob-fill {{ height: 10px; border-radius: 0 4px 4px 0; }}
.prob-value {{ text-align: right; color: {TEXT}; font-variant-numeric: tabular-nums; }}
@media (max-width: 760px) {{
    .specimen {{ grid-template-columns: 1fr; }}
    .specimen-photo {{ border-right: none; border-bottom: 1px solid {FIELD_LINE}; }}
    h1 {{ font-size: 2rem !important; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.logo(str(HERE / "assets" / "logo.svg"), size="large")

# --------------------------------------------------------------------------------------
# 2. Small helpers for calling the API
# --------------------------------------------------------------------------------------
# Every helper returns (data, error_message). Exactly one of them is None, so pages can do:
#     data, err = api_get("/health")
#     if err: show the error ... else: use data


def api_url() -> str:
    """The API base URL currently selected in the sidebar."""
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def _parse_error(resp: requests.Response) -> str:
    """FastAPI puts error text in {"detail": ...}; fall back to the raw body."""
    try:
        detail = resp.json().get("detail", resp.text)
        if isinstance(detail, dict):                 # structured errors, e.g. "not a maize leaf"
            detail = detail.get("message", detail)
        return f"{resp.status_code}: {detail}"
    except ValueError:
        return f"{resp.status_code}: {resp.text[:300]}"


def api_get(path: str, timeout: float = 10) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        resp = requests.get(f"{api_url()}{path}", timeout=timeout)
        if resp.ok:
            return resp.json(), None
        return None, _parse_error(resp)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {api_url()} ({type(exc).__name__})."


def api_post(path: str, files=None, data=None, timeout: float = 120) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        resp = requests.post(f"{api_url()}{path}", files=files, data=data, timeout=timeout)
        if resp.ok:
            return resp.json(), None
        return None, _parse_error(resp)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {api_url()} ({type(exc).__name__})."


# EDA statistics only change when the notebook is re-run, so cache them for 5 minutes.
@st.cache_data(ttl=300, show_spinner=False)
def get_eda_stats(base_url: str):
    try:
        resp = requests.get(f"{base_url}/eda-stats", timeout=10)
        return (resp.json(), None) if resp.ok else (None, _parse_error(resp))
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {base_url} ({type(exc).__name__})."


# --------------------------------------------------------------------------------------
# 3. Small display helpers
# --------------------------------------------------------------------------------------
def pct(value: Optional[float]) -> str:
    """0.9312 -> '93.1%'; None -> '—'."""
    return "—" if value is None else f"{value * 100:.1f}%"


def friendly_time(iso: Optional[str]) -> str:
    """ISO timestamp from the API -> '23 Nov 2026, 14:09 UTC'."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return iso


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    """Shared chart styling in the Ibigori palette: Figtree text, quiet field-line grid, no chart junk."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, system-ui, sans-serif", color=MUTED, size=13),
        title_font=dict(family="Literata, Georgia, serif", color=INK, size=17),
        hoverlabel=dict(font_size=13, font_family="Figtree, sans-serif", bgcolor=PAPER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=FIELD_LINE)
    fig.update_yaxes(gridwidth=1, gridcolor=FIELD_LINE, zeroline=False, linecolor=FIELD_LINE)
    return fig


def insight(text: str) -> None:
    """'What it tells us' note: a leaf-ink rule on a darker husk panel (instead of Streamlit's blue info box)."""
    text = html.escape(text).replace("_", " ")
    st.markdown(f'<div class="ib-insight"><b>What it tells us.</b> {text}</div>', unsafe_allow_html=True)


def page_header(title: str, intro: str) -> None:
    """Serif page title + one-sentence intro on the left, live model status on the right."""
    left, right = st.columns([4, 1.3])
    with left:
        st.title(title)
        st.markdown(f'<p class="ib-intro">{intro}</p>', unsafe_allow_html=True)
    with right:
        health, err = api_get("/health", timeout=4)
        if err:
            dot, text = LESION, "The model service is offline"
        elif health.get("model_loaded"):
            dot, text = "#2E8B57", f"Model <b>{health.get('model_version')}</b> is online"
        else:
            dot, text = SILK, "Service is up, model still loading"
        st.markdown(f'<div class="ib-status"><span class="ib-dot" style="background:{dot}"></span>{text}</div>',
                    unsafe_allow_html=True)


st.session_state.setdefault("api_url", DEFAULT_API_URL)


# ======================================================================================
# PAGE 1 — MODEL UPTIME
# ======================================================================================
def page_uptime():
    page_header("System health",
                "Is the model service up, how fast is it answering, and what is it busy with? "
                "This page refreshes itself every 5 seconds.")
    with st.expander("Connection settings"):
        st.text_input("Model service address (API URL)", key="api_url",
                      help="Where the FastAPI backend runs, e.g. http://localhost:8000 or your Azure URL.")

    # Keep a short history of health checks in the session to draw a response-time line.
    st.session_state.setdefault("health_history", [])

    # st.fragment(run_every=...) re-runs ONLY this function on a timer, not the whole page.
    @st.fragment(run_every=5)
    def live_health():
        start = time.perf_counter()
        health, err = api_get("/health", timeout=5)
        ping_ms = (time.perf_counter() - start) * 1000

        # Record this check (keep the last 60 = about 5 minutes)
        history = st.session_state["health_history"]
        history.append({"time": datetime.now(), "up": err is None,
                        "ping_ms": ping_ms if err is None else None})
        del history[:-60]

        if err:
            st.error(f"The model service is not answering. {err}")
            return

        # ---- Row 1: headline status -----------------------------------------------------
        status_ok = health["status"] == "ok"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", "Online" if status_ok else "Degraded")
        c2.metric("Uptime", health["uptime"])
        c3.metric("Model version", health.get("model_version") or "—")
        c4.metric("Health-check response", f"{ping_ms:.0f} ms")
        if health.get("startup_error"):
            st.warning(f"Model failed to load: {health['startup_error']}")
        st.caption(f"Service started {friendly_time(health['started_at'])}, "
                   f"model last trained {friendly_time(health.get('last_trained'))}")

        # ---- Row 2: model quality on the test set ----------------------------------------
        st.subheader("Model performance (test set)")
        m = health.get("model_metrics", {})
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", pct(m.get("accuracy")))
        k2.metric("Precision (macro)", pct(m.get("precision")))
        k3.metric("Recall (macro)", pct(m.get("recall")))
        k4.metric("F1-score (macro)", pct(m.get("f1_score")))
        k5.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "—")
        st.caption(f"Trained on {m.get('train_samples', '—')} images and tested on {m.get('test_samples', '—')} images "
                   "it never saw during training.")

        # ---- Row 3: traffic & latency ----------------------------------------------------
        st.subheader("Traffic & latency")
        p = health.get("prediction_stats", {})
        r = health.get("request_stats", {})
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Predictions served", p.get("total_predictions", 0))
        t2.metric("Avg prediction latency", f"{p['avg_latency_ms']:.0f} ms" if p.get("avg_latency_ms") else "—")
        t3.metric("p95 prediction latency", f"{p['p95_latency_ms']:.0f} ms" if p.get("p95_latency_ms") else "—")
        t4.metric("Low-confidence predictions", p.get("low_confidence_predictions", 0),
                  help=f"Confidence below {p.get('low_confidence_threshold', 0.65)} since the last retrain.")

        # "Is this a maize leaf?" guard: how many non-leaf images it has refused
        lc = health.get("leaf_check") or {}
        if lc:
            st.caption(f"Maize leaf check is **{lc.get('state')}**. It has refused {lc.get('rejected_images', 0)} "
                       f"picture(s) that were not maize leaves. {lc.get('message', '')}")

        left, right = st.columns([3, 2])
        with left:
            # Line chart of health-check response time during this session
            df = pd.DataFrame([h for h in history if h["up"]])
            if len(df) >= 2:
                fig = go.Figure(go.Scatter(
                    x=df["time"], y=df["ping_ms"], mode="lines+markers",
                    line=dict(width=2, color=INK), marker=dict(size=8, color=INK),
                    hovertemplate="%{x|%H:%M:%S}<br>%{y:.0f} ms<extra></extra>"))
                fig.update_layout(title="API response time (this session)", yaxis_title="ms")
                st.plotly_chart(style_figure(fig, 300), width="stretch")
            else:
                st.info("Collecting response times… the chart appears after a couple of refreshes.")
            uptime_pct = 100 * sum(h["up"] for h in history) / len(history)
            st.caption(f"Availability in this session: **{uptime_pct:.1f}%** of {len(history)} checks succeeded.")
        with right:
            # Requests per endpoint, as recorded by the API's middleware
            endpoints = r.get("endpoints", {})
            if endpoints:
                ep_df = (pd.DataFrame(endpoints).T.rename(columns={"count": "Requests", "avg_ms": "Avg ms"})
                         .sort_values("Requests", ascending=False))
                st.markdown(f"**Requests by endpoint**: {r.get('total_requests', 0)} in total, "
                            f"server errors {r.get('errors', 0)}")
                st.dataframe(ep_df, width="stretch")

        # ---- Row 4: retraining queue -----------------------------------------------------
        pending = health.get("pending_uploads", {})
        n_pending = sum(pending.values())
        st.subheader("Retraining queue")
        q1, q2 = st.columns([1, 3])
        q1.metric("New images waiting", n_pending)
        q2.write(f"Retraining is **{health.get('retraining', 'idle')}**. Waiting images: "
                 + (", ".join(f"{c}: {n}" for c, n in pending.items() if n) or "no uploads pending"))

    live_health()


# ======================================================================================
# PAGE 2 — VISUALIZATIONS
# ======================================================================================
def stats_box(per_class: Dict[str, Dict[str, float]], title: str, y_title: str, log_y: bool = False) -> go.Figure:
    """Box plot drawn from pre-computed statistics (the API sends quartiles, not raw pixels).
    Whiskers show the min and max of each class."""
    fig = go.Figure()
    for cls in CLASS_NAMES:
        s = per_class.get(cls)
        if not s:
            continue
        fig.add_trace(go.Box(
            x=[cls.replace("_", " ")], name=cls.replace("_", " "),
            q1=[s["q1"]], median=[s["median"]], q3=[s["q3"]],
            lowerfence=[s["min"]], upperfence=[s["max"]], mean=[s["mean"]],
            marker_color=CLASS_COLORS[cls], line=dict(width=2), boxmean=True,
        ))
    fig.update_layout(title=title, yaxis_title=y_title, showlegend=False)
    if log_y:
        fig.update_yaxes(type="log")
    return style_figure(fig)


def histogram_small_multiples(hist: Dict[str, Any], title: str, x_title: str) -> go.Figure:
    """One small histogram per class (side by side) from the pre-computed bin counts.
    Separate panels avoid four colours overlapping in one plot."""
    edges = hist["bin_edges"]
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    width = edges[1] - edges[0]
    if hist.get("log10"):  # texture bins were computed on log10(values) -> label with real values
        x_title += " (log₁₀)"
    fig = make_subplots(rows=1, cols=len(CLASS_NAMES), shared_yaxes=True,
                        subplot_titles=[c.replace("_", " ") for c in CLASS_NAMES], horizontal_spacing=0.03)
    for i, cls in enumerate(CLASS_NAMES, start=1):
        fig.add_trace(go.Bar(x=centers, y=hist["counts"][cls], width=width * 0.9,
                             marker_color=CLASS_COLORS[cls], name=cls,
                             hovertemplate=f"{cls}<br>bin %{{x:.2f}}<br>%{{y}} images<extra></extra>"),
                      row=1, col=i)
        fig.update_xaxes(title_text=x_title if i == 1 else None, row=1, col=i)
    fig.update_layout(title=title, showlegend=False)
    fig.update_yaxes(title_text="images", row=1, col=1)
    return style_figure(fig, 300)


def page_visualizations():
    page_header("Field data",
                "What the training leaves look like in numbers, and how well the model tells the diseases apart.")
    eda, err = get_eda_stats(api_url())
    if err:
        st.error(f"Could not load dataset statistics — {err}")
        return

    tab_data, tab_model = st.tabs(["Dataset features", "Model performance"])

    # ------------------------------------------------------------------ dataset features
    with tab_data:
        st.caption(f"Computed from {eda.get('n_train_images', '—')} training images, "
                   f"generated {friendly_time(eda.get('generated_at'))}")
        interp = eda.get("interpretations", {})
        feats = eda.get("features", {})

        # Feature 1 — class distribution
        st.subheader("How many leaves of each class")
        dist = eda["class_distribution"]
        dist_pct = eda.get("class_distribution_pct", {})
        fig = go.Figure(go.Bar(
            x=[c.replace("_", " ") for c in CLASS_NAMES], y=[dist.get(c, 0) for c in CLASS_NAMES],
            marker=dict(color=[CLASS_COLORS[c] for c in CLASS_NAMES], cornerradius=4),
            text=[f"{dist.get(c, 0)} ({dist_pct.get(c, 0):.1f}%)" for c in CLASS_NAMES], textposition="outside",
            hovertemplate="%{x}<br>%{y} images<extra></extra>"))
        fig.update_layout(title="Training images per class", yaxis_title="images",
                          yaxis_range=[0, max(dist.values()) * 1.2])
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(style_figure(fig), width="stretch")
        with c2:
            st.metric("Imbalance ratio (largest : smallest)", f"{eda.get('imbalance_ratio', 0):.2f} : 1")
            insight(interp.get('class_distribution', ''))

        # Feature 2 — chlorophyll vs necrosis colour index
        st.subheader("Green tissue versus dead tissue (mean green minus mean red)")
        ci = feats.get("color_index", {})
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(stats_box(ci.get("per_class", {}), "Colour index by class (box = middle 50%, ┼ = mean)",
                                  "Green − Red intensity"), width="stretch")
        with c2:
            insight(interp.get('color_index', ''))
            st.caption("Healthy tissue is green (high index). Dead or rusty tissue turns red-brown or tan, "
                       "so the index drops as disease spreads.")
        if ci.get("histogram"):
            with st.expander("Show the full distribution per class"):
                st.plotly_chart(histogram_small_multiples(ci["histogram"], "Colour index distribution",
                                                          "G − R"), width="stretch")

        # Feature 3 — lesion texture contrast
        st.subheader("Lesion texture (Laplacian variance)")
        tc = feats.get("texture_contrast", {})
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(stats_box(tc.get("per_class", {}), "Texture contrast by class (log scale)",
                                  "Laplacian variance", log_y=True), width="stretch")
        with c2:
            insight(interp.get('texture_contrast', ''))
            st.caption("The Laplacian highlights sharp changes in brightness: many small pustules or spots "
                       "give high values, smooth leaves give low values.")
        if tc.get("histogram"):
            with st.expander("Show the full distribution per class"):
                st.plotly_chart(histogram_small_multiples(tc["histogram"], "Texture contrast distribution",
                                                          "Laplacian variance"), width="stretch")

        # Supporting table: average colour per class
        if feats.get("mean_rgb"):
            with st.expander("Average pixel colour per class (table)"):
                rgb = pd.DataFrame(feats["mean_rgb"]).T[["red", "green", "blue"]].round(1)
                rgb.index = [i.replace("_", " ") for i in rgb.index]
                st.dataframe(rgb, width="stretch")

    # ------------------------------------------------------------------ model performance
    with tab_model:
        meta, err = api_get("/model-info")
        if err:
            st.error(f"Could not load model info — {err}")
            return
        st.caption(f"Model {meta.get('model_version', '—')}, trained {friendly_time(meta.get('last_trained'))}. "
                   f"{meta.get('architecture', '')}")
        k = st.columns(5)
        k[0].metric("Accuracy", pct(meta.get("accuracy")))
        k[1].metric("Precision (macro)", pct(meta.get("precision")))
        k[2].metric("Recall (macro)", pct(meta.get("recall")))
        k[3].metric("F1 (macro)", pct(meta.get("f1_score")))
        k[4].metric("ROC-AUC", f"{meta['roc_auc']:.3f}" if meta.get("roc_auc") is not None else "—")

        names = meta.get("class_names", CLASS_NAMES)
        c1, c2 = st.columns(2)

        # Confusion matrix: rows = true class, columns = predicted class.
        # Colour = share of that row (recall), text = raw count, so both are readable.
        cm = meta.get("confusion_matrix")
        if cm:
            cm_df = pd.DataFrame(cm, index=names, columns=names)
            row_share = cm_df.div(cm_df.sum(axis=1).replace(0, 1), axis=0)
            labels = [n.replace("_", " ") for n in names]
            fig = go.Figure(go.Heatmap(
                z=row_share.values, x=labels, y=labels, colorscale=SEQUENTIAL_GREENS, zmin=0, zmax=1,
                text=cm_df.values, texttemplate="%{text}", xgap=2, ygap=2,
                colorbar=dict(title="row share", tickformat=".0%"),
                hovertemplate="True %{y}<br>Predicted %{x}<br>%{text} images (%{z:.1%} of row)<extra></extra>"))
            fig.update_layout(title="Confusion matrix (counts; colour = % of true class)",
                              xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed")
            c1.plotly_chart(style_figure(fig, 420), width="stretch")

        # Per-class precision / recall / F1
        report = meta.get("per_class_report")
        if report:
            rows = [{"Class": c.replace("_", " "), "Precision": report[c]["precision"],
                     "Recall": report[c]["recall"], "F1": report[c]["f1-score"],
                     "Test images": int(report[c]["support"])} for c in names if c in report]
            rep_df = pd.DataFrame(rows)
            fig = go.Figure(go.Bar(
                y=rep_df["Class"], x=rep_df["F1"], orientation="h",
                marker=dict(color=[CLASS_COLORS.get(c.replace(" ", "_"), "#888") for c in rep_df["Class"]],
                            cornerradius=4),
                text=[f"{v:.1%}" for v in rep_df["F1"]], textposition="outside",
                hovertemplate="%{y}<br>F1 %{x:.1%}<extra></extra>"))
            fig.update_layout(title="F1-score per class", xaxis=dict(range=[0, 1.12], tickformat=".0%"))
            c2.plotly_chart(style_figure(fig, 300), width="stretch")
            c2.dataframe(rep_df.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}),
                         width="stretch", hide_index=True)


# ======================================================================================
# PAGE 3 — PREDICT ONE IMAGE
# ======================================================================================
# Scientific names for the specimen label (the fungus behind each disease; Zea mays for a healthy leaf)
LATIN = {
    "Blight": "Exserohilum turcicum",
    "Common_Rust": "Puccinia sorghi",
    "Gray_Leaf_Spot": "Cercospora zeae-maydis",
    "Healthy": "Zea mays",
}


def _sentence(text: str) -> str:
    """'its features are far away' -> 'Its features are far away.'"""
    text = (text or "").strip()
    return (text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")) if text else ""


def specimen_card(image_bytes: bytes, mime: str, filename: str,
                  result: Optional[Dict[str, Any]] = None, rejection: Optional[Dict[str, Any]] = None) -> str:
    """HTML for the result: the leaf photo mounted next to a printed herbarium-style label.
    Built as one HTML block so the photo and its label always sit together, like a filed specimen."""
    photo = f'<img src="data:{mime};base64,{base64.b64encode(image_bytes).decode()}" alt="Uploaded leaf photo">'
    examined = datetime.now().strftime("%d %b %Y, %H:%M")
    esc = html.escape

    if rejection is not None:                                   # the leaf check said "not a maize leaf"
        label = f"""
        <div class="label-frame">
          <div class="label-title">Specimen not accepted</div>
          <div class="label-reject">This doesn't look like a maize leaf</div>
          <p style="margin:10px 0 12px 0;font-family:Figtree,sans-serif;font-size:0.95rem;line-height:1.5">
            {esc(_sentence(rejection.get("reason", "it is too different from the maize leaves the model was trained on")))}
          </p>
          <table class="label-rows">
            <tr><td>File</td><td>{esc(filename)}</td></tr>
            <tr><td>Examined</td><td>{examined}</td></tr>
          </table>
          <div class="label-note">The model only knows maize leaves: northern leaf blight, common rust,
          gray leaf spot and healthy. It refuses to guess on other pictures rather than give a wrong
          diagnosis. Try a clear, close-up photo of one maize leaf.</div>
        </div>"""
    else:
        cls = result["predicted_class"]
        info = CLASS_INFO.get(cls, {"label": cls})
        probs = result["probabilities"]
        rows = "".join(
            f'<div class="prob-row"><span>{esc(CLASS_INFO.get(c, {}).get("label", c))}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{probs[c] * 100:.1f}%;'
            f'background:{CLASS_COLORS.get(c, MUTED)}"></div></div>'
            f'<span class="prob-value">{probs[c]:.1%}</span></div>'
            for c in sorted(probs, key=probs.get, reverse=True))
        note = ""
        if result.get("low_confidence"):
            note = ('<div class="label-note">The model is unsure about this leaf. The photo may be blurry, '
                    'show more than one problem, or look unlike the training images. Try another photo, or add '
                    'this one with the right label on the <i>Teach the model</i> page.</div>')
        label = f"""
        <div class="label-frame">
          <div class="label-title">Diagnosis</div>
          <div class="label-diagnosis">{esc(info["label"])}</div>
          <div class="label-latin">{esc(LATIN.get(cls, ""))}</div>
          <table class="label-rows">
            <tr><td>Confidence</td><td><span class="label-confidence">{result["confidence"]:.1%}</span></td></tr>
            <tr><td>Model</td><td>{esc(result["model_version"])}, custom CNN</td></tr>
            <tr><td>Examined</td><td>{examined}</td></tr>
            <tr><td>File</td><td>{esc(filename)}</td></tr>
          </table>
          {note}
          <div class="probs">{rows}</div>
        </div>"""
    card = f'<div class="specimen"><div class="specimen-photo">{photo}</div><div class="specimen-label">{label}</div></div>'
    # Collapse the indentation: Markdown would show indented HTML lines as a code block
    return " ".join(line.strip() for line in card.splitlines())


def page_predict():
    page_header("What's wrong with this maize leaf?",
                "Upload a photo of one maize leaf. You'll get the most likely disease, how sure the model is, "
                "and what to do next.")

    uploaded = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                help="A clear, close-up photo of a single leaf in daylight works best.")
    if not uploaded:
        st.caption("Tip: photograph the leaf flat, filling most of the picture, in daylight but out of harsh sun.")
        return

    image_bytes = uploaded.getvalue()
    mime = uploaded.type or "image/jpeg"
    with st.spinner("Examining the leaf…"):
        try:
            resp = requests.post(f"{api_url()}/predict", files={"file": (uploaded.name, image_bytes, mime)},
                                 timeout=120)
        except requests.RequestException as exc:
            st.error(f"Couldn't reach the model service at {api_url()} ({type(exc).__name__}). "
                     "Check that the API is running, then upload the photo again.")
            return

    if resp.status_code == 422:
        # The API's leaf check decided this is not a maize leaf, so no disease is predicted
        detail = resp.json().get("detail", {})
        st.markdown(specimen_card(image_bytes, mime, uploaded.name,
                                  rejection=detail if isinstance(detail, dict) else {"reason": str(detail)}),
                    unsafe_allow_html=True)
        return
    if not resp.ok:
        st.error(f"The model couldn't examine this photo: {_parse_error(resp)}")
        return

    result = resp.json()
    st.markdown(specimen_card(image_bytes, mime, uploaded.name, result=result), unsafe_allow_html=True)

    # What the result means for the farmer
    info = CLASS_INFO.get(result["predicted_class"], {"what": "", "action": ""})
    a, b = st.columns(2)
    with a:
        st.subheader("What it looks like")
        st.write(info["what"])
    with b:
        st.subheader("What to do")
        st.write(info["action"])
    st.caption(f"Answered in {result['latency_ms']:.0f} ms. This tool supports, but does not replace, advice from "
               "your local agricultural extension officer.")


# ======================================================================================
# PAGE 4 — UPLOAD DATA & RETRAIN
# ======================================================================================
STATUS_LABELS = {
    "pending": "Waiting for retraining",
    "in_training_set": "In training set",
    "duplicate": "Duplicate, skipped",
    "rejected": "Rejected",
}


def show_database():
    """Upload records and retraining runs straight from the database."""
    data, err = api_get("/uploads?limit=200")
    if err:
        st.error(err)
        return
    summary = data["summary"]
    by_status = summary.get("by_status", {})
    m = st.columns(5)
    m[0].metric("Upload records", summary.get("total", 0))
    m[1].metric("Waiting", by_status.get("pending", 0))
    m[2].metric("In training set", by_status.get("in_training_set", 0))
    m[3].metric("Duplicates", by_status.get("duplicate", 0))
    m[4].metric("Rejected", by_status.get("rejected", 0))

    tab_up, tab_runs = st.tabs(["Uploaded images", "Retraining runs"])
    with tab_up:
        rows = data["uploads"]
        if not rows:
            st.info("No uploads recorded yet.")
        else:
            df = pd.DataFrame(rows)
            df["status"] = df["status"].map(lambda v: STATUS_LABELS.get(v, v))
            df["uploaded_at"] = df["uploaded_at"].map(friendly_time)
            df = df.rename(columns={"uploaded_at": "Uploaded", "original_filename": "File", "label": "Label",
                                    "status": "Status", "reason": "Reason", "trained_into_version": "Learned by model",
                                    "batch_id": "Batch"})
            st.dataframe(df[["Uploaded", "File", "Label", "Status", "Reason", "Learned by model", "Batch"]],
                         width="stretch", hide_index=True, height=320)
    with tab_runs:
        runs, err = api_get("/retrain-runs")
        if err:
            st.error(err)
        elif not runs["runs"]:
            st.info("No retraining runs recorded yet.")
        else:
            df = pd.DataFrame(runs["runs"])
            df["started_at"] = df["started_at"].map(friendly_time)
            df["trigger"] = df["trigger"].fillna("").map(lambda t: t.split(":")[0])
            df = df.rename(columns={"id": "Run", "started_at": "Started", "trigger": "Trigger", "status": "Result",
                                    "from_version": "From", "to_version": "To",
                                    "images_from_uploads": "Uploaded images used", "accuracy_before": "Accuracy before",
                                    "accuracy_after": "Accuracy after"})
            st.dataframe(df[["Run", "Started", "Trigger", "Result", "From", "To", "Uploaded images used",
                             "Accuracy before", "Accuracy after"]].style.format(
                {"Accuracy before": "{:.1%}", "Accuracy after": "{:.1%}"}, na_rep="—"),
                width="stretch", hide_index=True)


def show_upload_result(res: Dict[str, Any]):
    """Summarise the /upload-data response: what was saved, what was rejected, did retraining start."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Images saved", res["saved"])
    c2.metric("Duplicates skipped", res["duplicates_skipped"])
    c3.metric("Rejected", len(res["rejected"]))
    if res.get("saved_per_class"):
        st.write("Saved per class: " + ", ".join(f"**{c.replace('_', ' ')}** {n}"
                                                 for c, n in res["saved_per_class"].items()))
    if res.get("batch_id"):
        st.caption(f"Every file of this upload is recorded in the database as batch `{res['batch_id']}` "
                   "(see section 3 below).")
    if res["rejected"]:
        with st.expander("Rejected files"):
            st.dataframe(pd.DataFrame(res["rejected"]), width="stretch", hide_index=True)
    trig = res.get("retrain_trigger", {})
    if trig.get("started"):
        st.success(f"**Retraining started automatically.** {trig['reason']}")
    else:
        st.info(trig.get("reason", ""))


def page_retrain():
    page_header("Teach the model",
                "Add photos of leaves whose disease you know. Once 10 or more are waiting, the model retrains "
                "itself on them, or you can start retraining yourself.")
        # ---------------------------------------------------------------- Step 1: upload
    st.header("1. Add labelled leaf photos")
    # Read the toggle BEFORE the tabs so both upload buttons below can use its value
    auto = st.toggle("Start retraining automatically when the trigger fires (10+ new images)", value=True)
    tab_zip, tab_imgs = st.tabs(["A zip with one folder per class", "Photos of one class"])

    with tab_zip:
        st.caption("A .zip containing one folder per class, e.g. `Blight/…jpg`, `Healthy/…jpg`. "
                   "The `demo_upload_batch.zip` from the notebook is already in this format.")
        zip_file = st.file_uploader("Zip file", type=["zip"], key="zip_upload")
        if st.button("Upload zip", type="primary", disabled=zip_file is None):
            with st.spinner("Uploading and checking images…"):
                res, err = api_post("/upload-data",
                                    files=[("files", (zip_file.name, zip_file.getvalue(), "application/zip"))],
                                    data={"auto_retrain": str(auto).lower()}, timeout=300)
            if err:
                st.error(err)
            else:
                show_upload_result(res)

    with tab_imgs:
        label = st.selectbox("These images are all…", CLASS_NAMES, format_func=lambda c: CLASS_INFO[c]["label"])
        imgs = st.file_uploader("Images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                accept_multiple_files=True, key="img_upload")
        if imgs:
            # Preview the first few images so the user can double-check the label
            st.image([i.getvalue() for i in imgs[:8]], width=90,
                     caption=[i.name[:14] for i in imgs[:8]])
            if len(imgs) > 8:
                st.caption(f"…and {len(imgs) - 8} more")
        if st.button(f"Upload {len(imgs) if imgs else 0} image(s) as {CLASS_INFO[label]['label']}",
                     type="primary", disabled=not imgs):
            files = [("files", (i.name, i.getvalue(), i.type or "image/jpeg")) for i in imgs]
            with st.spinner("Uploading and checking images…"):
                res, err = api_post("/upload-data", files=files,
                                    data={"label": label, "auto_retrain": str(auto).lower()}, timeout=300)
            if err:
                st.error(err)
            else:
                show_upload_result(res)

    # ---------------------------------------------------------------- Step 2: retrain
    st.header("2. Retrain the model")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Retrain now", type="primary", width="stretch"):
            res, err = api_post("/retrain", timeout=30)
            if err:
                st.error(err)
            else:
                st.success(f"Started — {res['pending_new_images']} new image(s) will be added.")
    with c2:
        st.caption("Retraining continues training the current model on old + new images, evaluates it on the fixed test "
                   "set, and only replaces the live model if accuracy does not drop by more than 2 points. "
                   "The API keeps answering predictions while it trains.")

    # Live status: this fragment re-runs every 3 seconds on its own
    @st.fragment(run_every=3)
    def live_status():
        status, err = api_get("/retrain-status", timeout=5)
        if err:
            st.error(err)
            return
        state = status["state"]
        dot = {"running": "#2a78d6", "completed": "#2E8B57", "rejected": SILK, "failed": LESION}.get(state, FIELD_LINE)
        st.markdown(f'<span class="ib-dot" style="background:{dot}"></span>Retraining is <b>{state}</b>. '
                    f"The live model is <b>{status.get('current_model_version', '—')}</b>.", unsafe_allow_html=True)
        if state == "running":
            total = max(status.get("total_epochs") or 1, 1)
            st.progress(min(status.get("epoch", 0) / total, 1.0),
                        text=f"Epoch {status.get('epoch', 0)}/{total} — {status.get('message', '')}")
        elif state == "completed":
            st.success(status["message"])
        elif state == "rejected":
            st.warning(status["message"])
        elif state == "failed":
            st.error(status["message"])
        else:
            st.caption(status.get("message", ""))

        # Before / after comparison for the most recent finished run
        result = status.get("result") or {}
        if state in ("completed", "rejected") and result.get("metrics"):
            before, after = result.get("previous_accuracy"), result["metrics"].get("accuracy")
            b1, b2, b3 = st.columns(3)
            b1.metric("Accuracy before", pct(before))
            b2.metric("Accuracy after", pct(after),
                      delta=f"{(after - before) * 100:+.1f} pts" if before is not None else None)
            b3.metric("New images used", sum(result.get("new_images", {}).values()))

    live_status()

    # ---------------------------------------------------------------- Step 3: database
    st.header("3. Database records")
    st.caption("Every uploaded image and every retraining run is stored in the SQLite database "
               "`data/pipeline.db` (`GET /uploads`, `GET /retrain-runs`).")
    show_database()

    # ---------------------------------------------------------------- Step 4: history
    st.header("4. Version history")
    meta, err = api_get("/model-info")
    if err:
        st.error(err)
        return
    history = meta.get("retrain_history", [])
    if not history:
        st.info(f"No retraining yet — the live model is the original **{meta.get('model_version', 'v1.0')}** "
                "from the notebook.")
        return

    rows = [{
        "When": friendly_time(run["timestamp"]),
        # Keep only the short part, e.g. "auto (upload)" instead of the full reason sentence
        "Trigger": run["trigger"].split(":")[0],
        "From": run["from_version"],
        "To": run["to_version"],
        "Result": "Promoted" if run["promoted"] else "Rejected, kept the old model",
        "New images": sum(run.get("new_images", {}).values()),
        "Accuracy before": run.get("previous_accuracy"),
        "Accuracy after": run["metrics"].get("accuracy"),
        "F1 after": run["metrics"].get("f1_score"),
    } for run in reversed(history)]
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Accuracy before": "{:.1%}", "Accuracy after": "{:.1%}", "F1 after": "{:.1%}"}, na_rep="—"),
        width="stretch", hide_index=True)

    # Accuracy across retraining runs (one line, so no legend needed)
    acc = [{"run": f"{r['from_version']}→{r['to_version']}", "accuracy": r["metrics"].get("accuracy"),
            "promoted": r["promoted"]} for r in history]
    fig = go.Figure(go.Scatter(
        x=[a["run"] for a in acc], y=[a["accuracy"] for a in acc], mode="lines+markers",
        line=dict(width=2, color=INK),
        marker=dict(size=10, color=INK, symbol=["circle" if a["promoted"] else "x" for a in acc]),
        hovertemplate="%{x}<br>test accuracy %{y:.1%}<extra></extra>"))
    fig.update_layout(title="Test accuracy after each retraining run (● promoted, ✕ rejected)",
                      yaxis=dict(tickformat=".0%", range=[0, 1.05]))
    st.plotly_chart(style_figure(fig, 300), width="stretch")


# --------------------------------------------------------------------------------------
# 5. Navigation in the top bar (no sidebar). The diagnosis page is the home page.
# --------------------------------------------------------------------------------------
pages = [
    st.Page(page_predict, title="Diagnose a leaf", icon=":material/eco:", default=True),
    st.Page(page_visualizations, title="Field data", icon=":material/insights:", url_path="field-data"),
    st.Page(page_retrain, title="Teach the model", icon=":material/school:", url_path="teach"),
    st.Page(page_uptime, title="System health", icon=":material/monitor_heart:", url_path="health"),
]
st.navigation(pages, position="top").run()
