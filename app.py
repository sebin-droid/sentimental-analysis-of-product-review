import streamlit as st

st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# IMPORTS  — every heavy package wrapped so the app shows a clear message
# --------------------------------------------------------------------------- #
import os
import re
import time
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except Exception as e:
    st.error(f"plotly import failed: {e}")
    st.stop()

try:
    import scipy.sparse as sp
except Exception as e:
    st.error(f"scipy import failed: {e}")
    st.stop()

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    for _p in ["vader_lexicon", "punkt", "stopwords", "punkt_tab"]:
        nltk.download(_p, quiet=True)
    _SIA = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"nltk import failed: {e}")
    st.stop()

TORCH_OK = False
try:
    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        pipeline as hf_pipeline,
    )
    TORCH_OK = True
except Exception:
    pass

# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #
DEVICE      = None
LABELS      = ["Negative", "Neutral", "Positive"]
I2L         = {0: "Negative", 1: "Neutral", 2: "Positive"}
L2I         = {"Negative": 0, "Neutral": 1, "Positive": 2}
CLR         = {"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"}

ASPECTS = {
    "Food / Product":  ["food","taste","flavor","menu","dish","meal","product","quality","item","fresh"],
    "Service":         ["service","staff","waiter","waitress","rude","friendly","helpful","server"],
    "Price / Value":   ["price","expensive","cheap","cost","value","worth","overpriced","affordable"],
    "Ambience":        ["ambience","atmosphere","decor","cozy","noisy","clean","dirty","vibe","location"],
    "Delivery / Speed":["delivery","fast","slow","wait","minutes","hours","shipping","quick","delay"],
    "Battery / Tech":  ["battery","camera","screen","performance","update","app","software","hardware"],
}

EMO_ICON = {"sadness":"😞","joy":"😄","love":"🥰",
            "anger":"U0001F621","fear":"U0001F628","surprise":"U0001F632"}

# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.hero{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460,#533483);
      border-radius:16px;padding:2.4rem 2rem 1.8rem;margin-bottom:1.6rem;
      text-align:center;box-shadow:0 8px 32px rgba(83,52,131,.4);}
.hero h1{color:#fff;font-size:2.4rem;font-weight:700;margin:0 0 .3rem;}
.hero p {color:rgba(255,255,255,.75);font-size:1rem;margin:0;}
.hero .pills{margin-top:.9rem;display:flex;justify-content:center;gap:.4rem;flex-wrap:wrap;}
.hero .pill{background:rgba(255,255,255,.12);color:#e2e8f0;padding:.2rem .75rem;
            border-radius:50px;font-size:.78rem;border:1px solid rgba(255,255,255,.2);}

.sh{font-size:1.05rem;font-weight:700;color:#1e293b;
    border-left:4px solid #6366f1;padding-left:.65rem;margin:1.4rem 0 .8rem;}

.badge{display:inline-flex;align-items:center;gap:.4rem;padding:.5rem 1.3rem;
       border-radius:50px;font-weight:700;font-size:1.1rem;
       box-shadow:0 4px 16px rgba(0,0,0,.15);}
.bp{background:linear-gradient(90deg,#43e97b,#38f9d7);color:#064e3b;}
.bn{background:linear-gradient(90deg,#f093fb,#f5576c);color:#fff;}
.bu{background:linear-gradient(90deg,#f6d365,#fda085);color:#451a03;}

.apills{display:flex;flex-wrap:wrap;gap:.45rem;margin:.5rem 0;}
.apill{padding:.28rem .85rem;border-radius:50px;font-size:.83rem;font-weight:600;border:2px solid;}
.ap{background:#dcfce7;border-color:#16a34a;color:#14532d;}
.an{background:#fee2e2;border-color:#dc2626;color:#7f1d1d;}
.au{background:#fef9c3;border-color:#ca8a04;color:#713f12;}

.fake{background:linear-gradient(90deg,#7f1d1d,#dc2626);color:#fff;
      padding:.9rem 1.3rem;border-radius:10px;font-weight:700;font-size:.95rem;}
.genuine{background:linear-gradient(90deg,#064e3b,#10b981);color:#fff;
         padding:.9rem 1.3rem;border-radius:10px;font-weight:700;font-size:.95rem;}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b);}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def clean(text):
    t = str(text).lower().strip()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"[^\w\s!?.,'\"-]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def hand_features(text):
    t = str(text); w = t.lower().split(); vs = _SIA.polarity_scores(t)
    return np.array([[
        vs["pos"], vs["neg"], vs["neu"], vs["compound"],
        len(w), len(t),
        t.count("!") / (len(w)+1),
        t.count("?") / (len(w)+1),
        sum(c.isupper() for c in t) / (len(t)+1),
        len(set(w)) / (len(w)+1),
        sum(1 for x in w if x in {"not","no","never","neither","nor","nothing","nobody"}),
        int(bool(re.search(r"http|www\.", t, re.I))),
        np.mean([len(x) for x in w]) if w else 0,
    ]], dtype=np.float32)

def absa(text):
    sents = re.split(r"[.!?;]", str(text))
    out = {}
    for asp, kws in ASPECTS.items():
        rel = [s for s in sents if any(k in s.lower() for k in kws)]
        if not rel:
            continue
        sc = float(np.mean([_SIA.polarity_scores(s)["compound"] for s in rel]))
        lbl = "Positive" if sc >= 0.05 else ("Negative" if sc <= -0.05 else "Neutral")
        out[asp] = {"score": round(sc, 3), "label": lbl}
    return out

# --------------------------------------------------------------------------- #
# MODEL LOADING
# --------------------------------------------------------------------------- #
def _load(path):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_classical():
    return {k: _load(v) for k, v in {
        "tfidf":  "tfidf_vectorizer.pkl",
        "scaler": "feature_scaler.pkl",
        "SVM":    "svm_model.pkl",
        "LR":     "lr_model.pkl",
        "RF":     "rf_model.pkl",
        "XGB":    "xgb_model.pkl",
    }.items()}

@st.cache_resource(show_spinner=False)
def load_bert():
    if not TORCH_OK:
        return None, None
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    mdl = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3, id2label=I2L, label2id=L2I)
    if os.path.exists("best_distilbert.pt"):
        mdl.load_state_dict(torch.load("best_distilbert.pt", map_location=_dev))
    mdl.to(_dev).eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_emotion():
    if not TORCH_OK:
        return None
    return hf_pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=True,
        device=(0 if torch.cuda.is_available() else -1),
    )

@st.cache_resource(show_spinner=False)
def load_fake():
    return _load("fake_review_clf.pkl")

# --------------------------------------------------------------------------- #
# PREDICT
# --------------------------------------------------------------------------- #
def pred_classical(text, arts, key):
    clf = arts.get(key); tfidf = arts.get("tfidf"); scaler = arts.get("scaler")
    if clf is None or tfidf is None:
        return None, None, None
    ct  = clean(text)
    tv  = tfidf.transform([ct])
    fv  = scaler.transform(hand_features(ct)) if scaler else hand_features(ct)
    X   = sp.hstack([tv, sp.csr_matrix(fv)])
    if key == "XGB":
        X = X.toarray()
    t0  = time.perf_counter()
    p   = clf.predict(X)[0]
    ms  = round((time.perf_counter() - t0) * 1000, 2)
    prob = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
    return I2L[p], prob, ms

def pred_bert(text, tok, mdl):
    if not TORCH_OK or tok is None or mdl is None:
        return None, None, None
    enc = tok(text[:512], return_tensors="pt", truncation=True,
              padding=True, max_length=128).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    t0  = time.perf_counter()
    with torch.no_grad():
        logits = mdl(**enc).logits
    ms   = round((time.perf_counter() - t0) * 1000, 2)
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return I2L[int(np.argmax(prob))], prob, ms

def pred_fake(text, clf, tfidf):
    if clf is None:
        return None, None
    ct = clean(text); vs = _SIA.polarity_scores(ct); w = ct.split()
    tmax = float(tfidf.transform([ct]).max()) if tfidf else 0.0
    X = np.array([[vs["pos"], vs["neg"], vs["compound"], len(w),
                   ct.count("!")/(len(w)+1),
                   sum(c.isupper() for c in text)/(len(text)+1),
                   len(set(w))/(len(w)+1),
                   sum(1 for x in w if x in {"not","no","never"}),
                   np.mean([len(x) for x in w]) if w else 0, tmax]], dtype=np.float32)
    prob = clf.predict_proba(X)[0]
    return ("Fake" if np.argmax(prob) == 1 else "Genuine"), prob

# --------------------------------------------------------------------------- #
# CHARTS
# --------------------------------------------------------------------------- #
def chart_probs(probs):
    fig = go.Figure(go.Bar(
        x=list(probs), y=LABELS, orientation="h",
        marker_color=[CLR[l] for l in LABELS],
        text=[f"{v*100:.1f}%" for v in probs], textposition="outside",
    ))
    fig.update_layout(title="Confidence Scores", height=200,
                      xaxis=dict(showticklabels=False, range=[0, 1.25]),
                      margin=dict(l=10, r=60, t=35, b=5),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def chart_radar(asp):
    if not asp:
        return None
    keys = list(asp.keys())
    vals = [(asp[k]["score"]+1)/2 for k in keys]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=keys+[keys[0]],
        fill="toself", line=dict(color="#6366f1", width=2.5),
        fillcolor="rgba(99,102,241,0.18)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                   tickvals=[0,.25,.5,.75,1],
                                   ticktext=["-1","-.5","0","+.5","+1"])),
        showlegend=False, height=330,
        margin=dict(l=55,r=55,t=45,b=35), title="Aspect Radar",
    )
    return fig

def chart_emotion(results):
    results = sorted(results, key=lambda x: -x["score"])
    labels  = [f"{EMO_ICON.get(r['label'],'')} {r['label'].title()}" for r in results]
    vals    = [r["score"] for r in results]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=1),
        text=[f"{v*100:.1f}%" for v in vals], textposition="outside",
    ))
    fig.update_layout(title="Emotion Probabilities", height=260,
                      xaxis=dict(showticklabels=False, range=[0,1.2]),
                      margin=dict(l=15,r=65,t=38,b=5),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def chart_compare(rows):
    df = pd.DataFrame(rows)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Accuracy","F1 Macro","ms / sample"))
    bc  = ["#3b82f6","#f59e0b","#10b981","#8b5cf6","#ef4444"]
    for i, col in enumerate(["acc","f1","ms"], 1):
        fig.add_trace(go.Bar(x=df["model"], y=df[col],
                             marker_color=bc[:len(df)],
                             text=[f"{v:.3f}" for v in df[col]],
                             textposition="outside",
                             showlegend=False), row=1, col=i)
    fig.update_layout(height=360, margin=dict(t=55,b=15))
    fig.update_xaxes(tickangle=-28)
    return fig

# --------------------------------------------------------------------------- #
# SIDEBAR
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## Configuration")
    model_opts = (["DistilBERT","SVM","LR","RF","XGB"]
                  if TORCH_OK else ["SVM","LR","RF","XGB"])
    primary = st.selectbox("Primary model", model_opts)
    st.markdown("---")
    do_emotion = st.checkbox("Emotion Detection",    value=True)
    do_absa    = st.checkbox("Aspect Analysis",      value=True)
    do_fake    = st.checkbox("Fake Review Check",    value=True)
    do_compare = st.checkbox("Compare All Models",   value=False)
    st.markdown("---")
    st.markdown("**Artifact status**")
    for path, name in [
        ("best_distilbert.pt","DistilBERT"),
        ("tfidf_vectorizer.pkl","TF-IDF"),
        ("svm_model.pkl","SVM"),
        ("lr_model.pkl","LR"),
        ("rf_model.pkl","RF"),
        ("xgb_model.pkl","XGBoost"),
        ("fake_review_clf.pkl","Fake clf"),
    ]:
        icon = "green" if os.path.exists(path) else "red"
        st.markdown(
            f"<span style='color:{'#10b981' if icon=='green' else '#ef4444'}'>"
            f"{'●' if icon=='green' else '○'}</span> {name}",
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# HERO
# --------------------------------------------------------------------------- #
st.markdown("""
<div class="hero">
  <h1>SentimentIQ</h1>
  <p>Advanced Review Intelligence — Classical ML &amp; Transformers</p>
  <div class="pills">
    <span class="pill">Sentiment</span>
    <span class="pill">Aspect Analysis</span>
    <span class="pill">Emotion Detection</span>
    <span class="pill">Fake Review Check</span>
    <span class="pill">Model Comparison</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# TABS
# --------------------------------------------------------------------------- #
t1, t2, t3 = st.tabs(["Single Review", "Batch Processing", "Model Insights"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with t1:
    LEFT, RIGHT = st.columns([1, 1], gap="large")

    EXAMPLES = {
        "-- pick example --": "",
        "Amazing phone (positive)":
            "Battery life is incredible — lasts all day. Performance is blazing fast. "
            "Camera could be better in low light but overall an amazing device, totally worth the price.",
        "Rude service (negative)":
            "Staff was incredibly rude and dismissive. Waited 45 minutes for food that arrived cold. "
            "Completely overpriced for such a terrible experience. Never returning.",
        "Mixed feelings (neutral)":
            "I expected better from a brand this popular. Quality is okay but seriously overpriced. "
            "Delivery was super fast though, and packaging was great.",
        "Suspicious review (fake signal)":
            "BEST PRODUCT EVER!!!! ABSOLUTELY AMAZING!!!! BUY IT NOW!!!! YOU WILL LOVE IT SO MUCH!!!!",
        "Restaurant visit":
            "Food was delicious — pasta cooked to perfection. Service was friendly. "
            "Location is cozy but gets noisy on weekends. Prices are fair. Will come back!",
    }

    with LEFT:
        st.markdown('<div class="sh">Enter Review</div>', unsafe_allow_html=True)
        ex = st.selectbox("Quick example (or type your own below)", list(EXAMPLES.keys()))
        default = EXAMPLES[ex] if ex != "-- pick example --" else ""
        txt = st.text_area(
            "✍️ Type or paste any review here",
            value=default,
            height=180,
            placeholder="e.g. The battery life is great but the camera is disappointing...",
            help="Type anything — your own review, a copied comment, anything you want to classify."
        )
        go_btn = st.button("🔍 Predict Sentiment", type="primary", use_container_width=True)
        st.caption(f"{len(txt.split())} words · {len(txt)} chars")
    with RIGHT:
        if go_btn and txt.strip():
            st.markdown('<div class="sh">Results</div>', unsafe_allow_html=True)

            with st.spinner("Running analysis…"):
                arts = load_classical()

                # sentiment
                if primary == "DistilBERT":
                    tok, mdl = load_bert()
                    sent, probs, ms = pred_bert(clean(txt), tok, mdl)
                else:
                    sent, probs, ms = pred_classical(txt, arts, primary)

                if sent is None:
                    st.warning("Model not loaded — place the .pkl file next to app.py")
                else:
                    bc = {"Positive":"bp","Neutral":"bu","Negative":"bn"}[sent]
                    ei = {"Positive":"😊","Neutral":"😐","Negative":"😞"}[sent]
                    st.markdown(
                        f"<div style='text-align:center;padding:.8rem 0'>"
                        f"<span class='badge {bc}'>{ei} {sent}</span>"
                        f"<p style='color:#64748b;margin-top:.5rem;font-size:.85rem'>"
                        f"Model: <b>{primary}</b> &nbsp;|&nbsp; {ms} ms</p></div>",
                        unsafe_allow_html=True)

                    if probs is not None:
                        st.plotly_chart(chart_probs(probs), use_container_width=True)

                # compare all
                if do_compare:
                    st.markdown('<div class="sh">All Models</div>', unsafe_allow_html=True)
                    tok2, mdl2 = load_bert() if TORCH_OK else (None, None)
                    cols = st.columns(len(model_opts))
                    for mk, col in zip(model_opts, cols):
                        if mk == "DistilBERT":
                            lb, _, t = pred_bert(clean(txt), tok2, mdl2)
                        else:
                            lb, _, t = pred_classical(txt, arts, mk)
                        col.metric(mk, lb or "N/A", f"{t} ms" if t else "")

            # absa
            if do_absa:
                st.markdown('<div class="sh">Aspect Analysis</div>', unsafe_allow_html=True)
                asp = absa(txt)
                if asp:
                    html = '<div class="apills">'
                    for a, r in asp.items():
                        c = {"Positive":"ap","Neutral":"au","Negative":"an"}[r["label"]]
                        i = {"Positive":"✅","Neutral":"➖","Negative":"❌"}[r["label"]]
                        html += f'<span class="apill {c}">{i} {a} ({r["score"]:+.2f})</span>'
                    html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)
                    rc = chart_radar(asp)
                    if rc:
                        st.plotly_chart(rc, use_container_width=True)
                else:
                    st.info("No specific aspects detected.")

            # emotion
            if do_emotion:
                st.markdown('<div class="sh">Emotion Detection</div>', unsafe_allow_html=True)
                if not TORCH_OK:
                    st.info("Emotion detection requires torch + transformers.")
                else:
                    try:
                        epipe = load_emotion()
                        res   = epipe(txt[:512])[0]
                        if isinstance(res, dict):
                            res = [res]
                        top  = max(res, key=lambda x: x["score"])
                        icon = EMO_ICON.get(top["label"], "")
                        st.markdown(
                            f"<div style='background:#f8fafc;border-left:4px solid #8b5cf6;"
                            f"padding:.8rem 1rem;border-radius:8px;margin-bottom:.6rem'>"
                            f"<b>Primary:</b> {icon} {top['label'].title()} "
                            f"({top['score']*100:.1f}%)</div>",
                            unsafe_allow_html=True)
                        st.plotly_chart(chart_emotion(res), use_container_width=True)
                    except Exception as ex:
                        st.warning(f"Emotion model error: {ex}")

            # fake
            if do_fake:
                st.markdown('<div class="sh">Authenticity Check</div>', unsafe_allow_html=True)
                clf_f = load_fake()
                verdict, fp = pred_fake(txt, clf_f, arts.get("tfidf"))
                if verdict:
                    css = "fake" if verdict == "Fake" else "genuine"
                    icon = "⚠️" if verdict == "Fake" else "✅"
                    conf = fp[1] if verdict == "Fake" else fp[0]
                    st.markdown(
                        f'<div class="{css}">{icon} {verdict.upper()} '
                        f'— confidence {conf*100:.1f}%</div>',
                        unsafe_allow_html=True)
                else:
                    st.info("Fake classifier not loaded.")

        elif go_btn:
            st.warning("Please enter a review first.")

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="sh">Batch Processing</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with a **text** column (also: `review`, `comment`).")
    up = st.file_uploader("CSV file", type=["csv"])

    if up:
        df_up = pd.read_csv(up)
        tcol  = next((c for c in df_up.columns
                      if c.lower() in ["text","review","review_text","comment","body"]), None)
        if tcol is None:
            st.error("No text column found — rename it to `text`.")
        else:
            st.success(f"{len(df_up):,} reviews found in column `{tcol}`")
            n = st.slider("Reviews to process", 10, min(500, len(df_up)), 100, step=10)
            mk = st.selectbox("Model", ["SVM","LR","RF"] if not TORCH_OK
                              else ["SVM","LR","RF","DistilBERT"])

            if st.button("Run batch", type="primary"):
                texts   = df_up[tcol].fillna("").astype(str).head(n).tolist()
                arts_b  = load_classical()
                tok_b, mdl_b = (load_bert() if mk == "DistilBERT" else (None, None))
                prog    = st.progress(0.0)
                out     = []

                for i, t in enumerate(texts):
                    if mk == "DistilBERT":
                        lb, _, _ = pred_bert(clean(t), tok_b, mdl_b)
                    else:
                        lb, _, _ = pred_classical(t, arts_b, mk)
                    out.append({"review": t[:80]+"…", "sentiment": lb or "N/A"})
                    prog.progress((i+1)/len(texts))

                res = pd.DataFrame(out)
                vc  = res["sentiment"].value_counts()
                c1,c2,c3 = st.columns(3)
                c1.metric("Positive", int(vc.get("Positive",0)))
                c2.metric("Neutral",  int(vc.get("Neutral",0)))
                c3.metric("Negative", int(vc.get("Negative",0)))

                fig_d = go.Figure(go.Pie(
                    labels=vc.index, values=vc.values, hole=0.5,
                    marker=dict(colors=[CLR.get(l,"#94a3b8") for l in vc.index],
                                line=dict(color="white",width=2))))
                fig_d.update_layout(title=f"Distribution — {mk}", height=300,
                                    margin=dict(t=40,b=5))
                st.plotly_chart(fig_d, use_container_width=True)
                st.dataframe(res, use_container_width=True)
                st.download_button("Download CSV", res.to_csv(index=False),
                                   "results.csv", "text/csv")

# ── TAB 3 ────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="sh">Model Performance</div>', unsafe_allow_html=True)

    perf = [
        {"model":"SVM",     "acc":0.76,"f1":0.73,"ms":0.05,"size":"12 MB"},
        {"model":"LR",      "acc":0.75,"f1":0.72,"ms":0.08,"size":"4 MB"},
        {"model":"RF",      "acc":0.71,"f1":0.68,"ms":1.20,"size":"55 MB"},
        {"model":"XGBoost", "acc":0.74,"f1":0.71,"ms":0.85,"size":"18 MB"},
        {"model":"DistilBERT","acc":0.88,"f1":0.86,"ms":22.0,"size":"255 MB"},
    ]
    st.plotly_chart(chart_compare(perf), use_container_width=True)
    st.dataframe(pd.DataFrame(perf).set_index("model"), use_container_width=True)

    st.markdown('<div class="sh">Architecture</div>', unsafe_allow_html=True)
    st.code("""
Input Review
     |
     +-- Text Cleaning (lowercase, strip URLs/HTML)
     |
     +-- Feature Engineering (13 hand-crafted features)
     |     VADER scores, word/char count, exclamation density,
     |     caps ratio, unique word ratio, negation count ...
     |
     +--[A] TF-IDF (50k unigrams+bigrams) + Scaled Features
     |        |-- SVM (LinearSVC)
     |        |-- Logistic Regression
     |        |-- Random Forest
     |        +-- XGBoost
     |
     +--[B] DistilBERT fine-tuned (3 epochs, weighted CE loss)
     |
     +--[C] GoEmotions pipeline  --> joy/anger/sadness/fear/love/surprise
     |
     +--[D] ABSA (VADER + aspect keywords) --> per-aspect sentiment
     |
     +--[E] GradientBoosting fake detector --> Genuine / Fake
""", language="text")

    st.markdown('<div class="sh">How to run</div>', unsafe_allow_html=True)
    st.info("""
1. Run `01_EDA__1_.ipynb` → produces `preprocessed_data.csv`
2. Run `modeling_sentimemntal_analysis_.ipynb` → downloads all `.pkl` + `.pt` files
3. Place all artifacts in the same folder as `app.py`
4. `pip install -r requirements.txt`
5. `streamlit run app.py`
    """)
