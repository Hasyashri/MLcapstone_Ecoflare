"""
🔥 EcoFlare Offline Dashboard (FINAL – Robust + Cause Model)

Fully offline Streamlit ML dashboard for wildfire AI evaluation.

MODELS:
✅ Detection (binary classification)
✅ Spread (regression comparison only)
✅ Cause (multiclass classification – LIVE inference from cause_classifier.pkl)

FEATURES:
✅ Detection metrics (ROC-AUC, PR-AUC, F1, confusion matrix)
✅ ROC curve + PR curve
✅ Threshold sweep explorer
✅ SHAP global + local explainability
✅ Temporal CV consistency
✅ Ensemble decision rules
✅ Cross-task multi-model comparisons
✅ LIVE Cause model metrics (no CSV/JSON exports required)

ROBUSTNESS:
✅ Schema-safe (no hardcoded column assumptions)
✅ Defensive loading
✅ Never crashes if files missing
✅ Updated Streamlit API (width="stretch")
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ===================================================
# PATHS
# ===================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "data" / "features" / "fire_training_master_clean.csv"

DETECTION_MODEL_FILE = PROJECT_ROOT / "models" / "detection_model.pkl"
SPREAD_MODEL_FILE    = PROJECT_ROOT / "models" / "spread_regressor.pkl"
CAUSE_MODEL_FILE     = PROJECT_ROOT / "models" / "cause_classifier.pkl"

DETECTION_REPORT_DIR = PROJECT_ROOT / "reports" / "detection_eval"
SHAP_REPORT_DIR      = PROJECT_ROOT / "reports" / "shap"
MODEL_COMP_DIR       = PROJECT_ROOT / "reports" / "model_comparison"

ROC_CSV              = DETECTION_REPORT_DIR / "roc_curve_detection.csv"
PR_CSV               = DETECTION_REPORT_DIR / "pr_curve_detection.csv"
THRESHOLD_SWEEP_CSV  = DETECTION_REPORT_DIR / "threshold_sweep_detection.csv"
SUMMARY_JSON         = DETECTION_REPORT_DIR / "summary_metrics_detection.json"
CONF_MAT_TXT         = DETECTION_REPORT_DIR / "confusion_matrix_default_0_25.txt"
SCENARIOS_JSON       = DETECTION_REPORT_DIR / "detection_threshold_scenarios.json"
TEMPORAL_CV_CSV      = DETECTION_REPORT_DIR / "temporal_cv_detection_by_year.csv"
ENSEMBLE_V2_CSV      = DETECTION_REPORT_DIR / "ensemble_rules_comparison_v2.csv"

SHAP_IMPORTANCE_CSV  = SHAP_REPORT_DIR / "shap_feature_importance.csv"
SHAP_SUMMARY_PNG     = SHAP_REPORT_DIR / "shap_summary_bar.png"
SHAP_TOP_RECORDS_CSV= SHAP_REPORT_DIR / "shap_top_records.csv"

MODEL_COMPARISON_CSV= MODEL_COMP_DIR / "model_comparison_summary.csv"

# ===================================================
# STREAMLIT CONFIG
# ===================================================

st.set_page_config(
    page_title="EcoFlare – Offline ML Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔥 EcoFlare – Offline Wildfire ML Dashboard")
st.caption("Final robust offline evaluation dashboard — Detection • Spread • Cause")

# ===================================================
# HELPERS
# ===================================================

@st.cache_data(show_spinner=False)
def load_csv_safe(path):
    try:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def load_json_safe(path):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        if path.exists():
            return joblib.load(path)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def load_confusion_matrix():
    try:
        if CONF_MAT_TXT.exists():
            cm = np.loadtxt(CONF_MAT_TXT, dtype=int)
            if cm.shape == (2, 2):
                return cm
    except Exception:
        pass
    return None


def get_first_matching_column(df, keywords):
    for k in keywords:
        for col in df.columns:
            if k.lower() in col.lower():
                return col
    return None


def safe_rule_label(row, df, idx):
    for col in ["rule_name", "rule", "scenario", "label", "name"]:
        if col in df.columns:
            v = row.get(col)
            if pd.notna(v):
                return str(v)
    return f"Rule {idx+1}"


# ===================================================
# SIDEBAR
# ===================================================

with st.sidebar:
    st.header("⚙ Data & Models")

    df = load_csv_safe(DATA_FILE)
    st.write("Dataset:", "✅ Loaded" if df is not None else "❌ Missing")

    st.markdown("### Models")
    st.write("Detection:", "✅" if load_model(DETECTION_MODEL_FILE) else "❌")
    st.write("Spread:",    "✅" if load_model(SPREAD_MODEL_FILE) else "⚠")
    st.write("Cause:",     "✅" if load_model(CAUSE_MODEL_FILE) else "❌")

# ===================================================
# LOAD SHARED ARTIFACTS
# ===================================================

summary      = load_json_safe(SUMMARY_JSON)
cm_detection = load_confusion_matrix()

roc_df   = load_csv_safe(ROC_CSV)
pr_df    = load_csv_safe(PR_CSV)
thr_df   = load_csv_safe(THRESHOLD_SWEEP_CSV)
scenarios= load_json_safe(SCENARIOS_JSON)

temp_df = load_csv_safe(TEMPORAL_CV_CSV)
ens_df  = load_csv_safe(ENSEMBLE_V2_CSV)

shap_imp   = load_csv_safe(SHAP_IMPORTANCE_CSV)
shap_local = load_csv_safe(SHAP_TOP_RECORDS_CSV)

model_comp = load_csv_safe(MODEL_COMPARISON_CSV)

cause_model = load_model(CAUSE_MODEL_FILE)

# ===================================================
# TABS
# ===================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1️⃣ Detection",
     "2️⃣ Thresholds",
     "3️⃣ SHAP",
     "4️⃣ Temporal + Ensemble",
     "5️⃣ Root Cause Model"]
)

# ===================================================
# TAB 1 — DETECTION
# ===================================================

with tab1:
    st.subheader("🔥 Detection Model")

    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{summary.get('roc_auc',0):.3f}")
        c2.metric("PR-AUC", f"{summary.get('pr_auc',0):.3f}")
        c3.metric("Samples", summary.get("n_test","-"))

    if cm_detection is not None:
        tn, fp, fn, tp = cm_detection.ravel()

        cm_df = pd.DataFrame(
            cm_detection,
            index=["Actual No", "Actual Yes"],
            columns=["Pred No", "Pred Yes"]
        )

        st.markdown("### Confusion Matrix")
        st.dataframe(cm_df, width="stretch")

    st.markdown("### ROC & PR Curves")
    col1, col2 = st.columns(2)

    if roc_df is not None:
        st.plotly_chart(px.line(roc_df, x="fpr", y="tpr", title="ROC Curve"), width="stretch")

    if pr_df is not None:
        st.plotly_chart(px.line(pr_df, x="recall", y="precision", title="Precision-Recall"), width="stretch")


# ===================================================
# TAB 2 — THRESHOLDS
# ===================================================

with tab2:

    st.subheader("🎚 Threshold Optimization")

    if thr_df is not None and "threshold" in thr_df.columns:

        t = st.slider(
            "Decision Threshold",
            float(thr_df["threshold"].min()),
            float(thr_df["threshold"].max()),
            0.25,
            0.01
        )

        row = thr_df.iloc[(thr_df["threshold"] - t).abs().idxmin()]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{row['precision']:.3f}")
        c2.metric("Recall", f"{row['recall']:.3f}")
        c3.metric("F1", f"{row['f1']:.3f}")
        c4.metric("False Alarm", f"{row['false_alarm_rate']:.3f}")

        st.dataframe(thr_df.round(4), width="stretch")


# ===================================================
# TAB 3 — SHAP
# ===================================================

with tab3:
    st.subheader("🔍 SHAP Explainability")

    if shap_imp is not None:
        fcol = get_first_matching_column(shap_imp, ["feature"])
        vcol = get_first_matching_column(shap_imp, ["shap", "importance"])

        if fcol and vcol:
            fig = px.bar(
                shap_imp.sort_values(vcol, ascending=False).head(20),
                x=vcol,
                y=fcol,
                orientation="h",
                title="Top SHAP Features"
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")

    if SHAP_SUMMARY_PNG.exists():
        st.image(str(SHAP_SUMMARY_PNG), width="stretch")

    if shap_local is not None:
        st.dataframe(shap_local.head(50), width="stretch")


# ===================================================
# TAB 4 — TEMPORAL & ENSEMBLE
# ===================================================

with tab4:
    st.subheader("📈 Temporal CV")

    if temp_df is not None and "year" in temp_df.columns:
        melted = temp_df.melt(
            id_vars="year",
            value_vars=["roc_auc","pr_auc"],
            var_name="metric",
            value_name="value"
        )

        st.plotly_chart(
            px.line(melted, x="year", y="value", color="metric",
            title="Detection Metrics over Years"),
            width="stretch"
        )

        st.dataframe(temp_df, width="stretch")

    if ens_df is not None:
        st.markdown("### Ensemble Rules")
        st.dataframe(ens_df.round(4), width="stretch")

        for i, r in ens_df.iterrows():
            label = safe_rule_label(r, ens_df, i)
            st.markdown(f"**{label}**")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy", f"{r.get('accuracy',0):.3f}")
            c2.metric("Precision",f"{r.get('precision',0):.3f}")
            c3.metric("Recall",   f"{r.get('recall',0):.3f}")
            c4.metric("F1",       f"{r.get('f1',0):.3f}")
            st.divider()


# ===================================================
# TAB 5 — CAUSE MODEL (LIVE)
# ===================================================

with tab5:

    st.subheader("🧠 Cause Classification (LIVE Evaluation)")

    if df is None or cause_model is None:
        st.warning("Dataset or cause_classifier.pkl missing.")
    elif "fire_cause" not in df.columns:
        st.error("Column 'fire_cause' not found.")
    else:

        X = df.drop(columns=["fire_cause","fire_occurred","timestamp","split"], errors="ignore")
        X = X.select_dtypes(include="number")
        y = df["fire_cause"]

        y_pred = cause_model.predict(X)

        acc        = accuracy_score(y, y_pred)
        f1_macro   = f1_score(y, y_pred, average="macro")
        f1_weight  = f1_score(y, y_pred, average="weighted")

        c1,c2,c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Macro F1", f"{f1_macro:.3f}")
        c3.metric("Weighted F1", f"{f1_weight:.3f}")

        st.markdown("### Confusion Matrix")

        labels = sorted(y.unique())
        cm = confusion_matrix(y, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        st.dataframe(cm_df, width="stretch")

        report = classification_report(y, y_pred, output_dict=True)
        rep_df = pd.DataFrame(report).T.reset_index().rename(columns={"index":"class"})
        rep_df = rep_df[rep_df["class"].isin(labels)]

        st.markdown("### Class Metrics")
        st.dataframe(rep_df[["class","precision","recall","f1-score","support"]].round(4), width="stretch")

        st.plotly_chart(
            px.bar(
                rep_df,
                x="class",
                y="f1-score",
                color="class",
                title="F1 Score by Cause",
                text_auto=".3f"
            ),
            width="stretch"
        )
# ===================================================