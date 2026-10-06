import os, joblib, pandas as pd, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import glob
import streamlit as st
import matplotlib.pyplot as plt
from train import add_features

st.set_page_config(page_title="Bank Churn Risk Scoring", page_icon="🏦", layout="wide")

def find(name):
    hits = glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None

@st.cache_resource
def load():
    err, mp = None, find("best_model.joblib")
    if mp:
        try:
            return joblib.load(mp)
        except Exception as e:
            err = repr(e)
    csv = find("Churn_Modelling.csv") or find("European_Bank.csv")
    if csv:
        import train
        train.DATA = csv; train.OUT = "models"
        train.main()
        return joblib.load("models/best_model.joblib")
    st.error("Could not load the model and no dataset was found to retrain.")
    if err: st.code(f"Model load error: {err}")
    st.write("Files the app can see in the repo:")
    st.code("\n".join(sorted(os.path.join(r, f) for r, _, fs in os.walk(".") for f in fs
                           if ".git" not in r and "__pycache__" not in r)))
    st.stop()

art = load(); pipe = art["pipe"]
metrics = pd.read_csv(find("metrics.csv"))
imp = pd.read_csv(find("feature_importance.csv"))
scores = pd.read_csv(find("test_scores.csv"))

def score(d):
    return float(pipe.predict_proba(add_features(pd.DataFrame([d])))[:, 1][0])

st.title("🏦 Customer Churn Risk Scoring System")
st.caption(f"Best model: **{art['name']}** (selected by ROC-AUC on a stratified hold-out set)")

with st.sidebar:
    st.header("Customer profile")
    c = dict(
        CreditScore=st.slider("Credit score", 300, 900, 650),
        Geography=st.selectbox("Geography", ["France", "Spain", "Germany"]),
        Gender=st.selectbox("Gender", ["Female", "Male"]),
        Age=st.slider("Age", 18, 92, 40),
        Tenure=st.slider("Tenure (years)", 0, 10, 5),
        Balance=st.number_input("Balance", 0.0, 300000.0, 75000.0, 1000.0),
        NumOfProducts=st.slider("Number of products", 1, 4, 2),
        HasCrCard=int(st.checkbox("Has credit card", True)),
        IsActiveMember=int(st.checkbox("Active member", True)),
        EstimatedSalary=st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, 1000.0),
    )
    thr = st.slider("Churn threshold", 0.05, 0.95, 0.50, 0.05)

t1, t2, t3, t4, t5 = st.tabs(["Risk calculator", "Probability distribution",
                              "Feature importance", "What-if simulator", "Model comparison"])

with t1:
    p = score(c)
    a, b, d = st.columns(3)
    a.metric("Churn probability", f"{p:.1%}")
    b.metric("Binary flag", "⚠️ Will churn" if p >= thr else "✅ Likely to stay")
    d.metric("Risk band", "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low")
    st.progress(min(p, 1.0))

with t2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[scores.actual == 0].prob, bins=30, alpha=.6, label="Retained")
    ax.hist(scores[scores.actual == 1].prob, bins=30, alpha=.6, label="Churned")
    ax.axvline(score(c), color="k", ls="--", label="This customer")
    ax.axvline(thr, color="r", ls=":", label="Threshold")
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Customers"); ax.legend()
    st.pyplot(fig)

with t3:
    top = imp.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.Feature, top.Importance); ax.set_xlabel("Drop in ROC-AUC when shuffled")
    st.pyplot(fig)
    st.dataframe(imp, width="stretch")
    if find("shap_summary.png"):
        st.subheader("SHAP summary"); st.image(find("shap_summary.png"))
    if find("partial_dependence.png"):
        st.subheader("Partial dependence"); st.image(find("partial_dependence.png"))

with t4:
    st.write("Change engagement / product values and compare against the profile in the sidebar.")
    x1, x2, x3, x4 = st.columns(4)
    s = dict(c)
    s["NumOfProducts"] = x1.slider("Products", 1, 4, c["NumOfProducts"], key="w1")
    s["IsActiveMember"] = int(x2.checkbox("Active", bool(c["IsActiveMember"]), key="w2"))
    s["Balance"] = x3.number_input("Balance", 0.0, 300000.0, float(c["Balance"]), 1000.0, key="w3")
    s["Tenure"] = x4.slider("Tenure", 0, 10, c["Tenure"], key="w4")
    p0, p1 = score(c), score(s)
    m1, m2 = st.columns(2)
    m1.metric("Baseline", f"{p0:.1%}")
    m2.metric("Scenario", f"{p1:.1%}", f"{(p1 - p0) * 100:+.1f} pts", delta_color="inverse")
    var = st.selectbox("Sweep a variable", ["Age", "NumOfProducts", "Tenure", "Balance", "CreditScore"])
    rng = {"Age": range(18, 80, 2), "NumOfProducts": range(1, 5), "Tenure": range(0, 11),
           "Balance": range(0, 250001, 10000), "CreditScore": range(350, 851, 25)}[var]
    ys = [score({**s, var: v}) for v in rng]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(list(rng), ys, marker="o"); ax.set_xlabel(var); ax.set_ylabel("Churn probability")
    st.pyplot(fig)

with t5:
    st.dataframe(metrics.style.format({k: "{:.3f}" for k in metrics.columns[1:]}), width="stretch")
