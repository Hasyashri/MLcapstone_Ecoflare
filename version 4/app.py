"""
🔥 EcoFlare – Offline Wildfire ML Dashboard
FINAL – Detection • Root Cause • Spread • Maps

Fully offline dashboard that reads:
- Precomputed evaluation artifacts for DETECTION
- Trained models for:
    • detection_model.pkl      (XGBoost + calibration)
    • cause_classifier.pkl     (LightGBM multiclass)
    • spread_regressor.pkl     (LightGBM regression)

Tabs:
1️⃣ Overview (status + quick model summary)
2️⃣ Detection Metrics (JSON + confusion + curves + thresholds)
3️⃣ SHAP Explainability
4️⃣ Temporal CV + Ensemble Rules
5️⃣ Root Cause & Spread Performance
6️⃣ Risk Map (sample points with detection probability)

NOTE:
- This app assumes `fire_training_master_clean.csv`
  is the same schema used in Steps 7, 8, and 9.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional imports for LightGBM models and map
try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False


# ===================================================
# PATHS
# ===================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "data" / "features" / "fire_training_master_clean.csv"

DETECTION_MODEL_FILE = PROJECT_ROOT / "models" / "detection_model.pkl"
CAUSE_MODEL_FILE     = PROJECT_ROOT / "models" / "cause_classifier.pkl"
SPREAD_MODEL_FILE    = PROJECT_ROOT / "models" / "spread_regressor.pkl"

DETECTION_REPORT_DIR = PROJECT_ROOT / "reports" / "detection_eval"
SHAP_REPORT_DIR      = PROJECT_ROOT / "reports" / "shap"
MODEL_COMP_DIR       = PROJECT_ROOT / "reports" / "model_comparison"

# Detection artifacts
ROC_CSV             = DETECTION_REPORT_DIR / "roc_curve_detection.csv"
PR_CSV              = DETECTION_REPORT_DIR / "pr_curve_detection.csv"
THRESHOLD_SWEEP_CSV = DETECTION_REPORT_DIR / "threshold_sweep_detection.csv"
SUMMARY_JSON        = DETECTION_REPORT_DIR / "summary_metrics_detection.json"
CONF_MAT_TXT        = DETECTION_REPORT_DIR / "confusion_matrix_default_0_25.txt"
SCENARIOS_JSON      = DETECTION_REPORT_DIR / "detection_threshold_scenarios.json"
TEMPORAL_CV_CSV     = DETECTION_REPORT_DIR / "temporal_cv_detection_by_year.csv"
ENSEMBLE_V2_CSV     = DETECTION_REPORT_DIR / "ensemble_rules_comparison_v2.csv"

# SHAP artifacts
SHAP_IMPORTANCE_CSV = SHAP_REPORT_DIR / "shap_feature_importance.csv"
SHAP_SUMMARY_PNG    = SHAP_REPORT_DIR / "shap_summary_bar.png"
SHAP_TOP_RECORDS_CSV = SHAP_REPORT_DIR / "shap_top_records.csv"

# Model comparison (optional)
MODEL_COMPARISON_CSV = MODEL_COMP_DIR / "model_comparison_summary.csv"


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
st.caption("Final robust offline evaluation – Detection • Root Cause • Spread • Map")


# ===================================================
# HELPERS
# ===================================================

@st.cache_data(show_spinner=False)
def load_csv_safe(path: Path):
    try:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def load_json_safe(path: Path):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def load_dataset():
    return load_csv_safe(DATA_FILE)


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
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


def get_first_matching_column(df: pd.DataFrame, patterns):
    """Return first column whose lowercase name contains any of the patterns."""
    for p in patterns:
        for c in df.columns:
            if p in c.lower():
                return c
    return None


def pick_metric_columns(df: pd.DataFrame, candidates):
    cols = []
    for c in candidates:
        if c in df.columns and not df[c].isna().all():
            cols.append(c)
    return cols


def safe_rule_label(row: pd.Series, df: pd.DataFrame, idx: int) -> str:
    """Label for ensemble rules without hardcoding only 'rule_name'."""
    for col in ["rule_name", "rule", "scenario", "label", "name"]:
        if col in df.columns:
            val = row.get(col, None)
            if pd.notna(val):
                return str(val)
    return f"Rule {idx + 1}"


# ===== Feature builders aligned with training scripts =====

def build_features_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7 (detection):
    X = df.drop(columns=[TARGET, "fire_cause", "timestamp", "split"], errors="ignore")
    X = X.select_dtypes(include="number")
    TARGET = 'fire_occurred'
    """
    drop_cols = ["fire_occurred", "fire_cause", "timestamp", "split"]
    X = df.drop(columns=drop_cols, errors="ignore")
    return X.select_dtypes(include="number")


def build_features_cause(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 8 (cause):
    X = df.drop(columns=[TARGET_COL, "fire_occurred", "timestamp", "split"], errors="ignore")
    X = X.select_dtypes(include="number")
    TARGET_COL = 'fire_cause'
    """
    drop_cols = ["fire_cause", "fire_occurred", "timestamp", "split"]
    X = df.drop(columns=drop_cols, errors="ignore")
    return X.select_dtypes(include="number")


def build_features_spread(df_fire: pd.DataFrame) -> pd.DataFrame:
    """
    Step 9 (spread):
    DROP_COLS = ["fire_occurred", "fire_cause", "timestamp", "split"]
    X = df_fire.drop(columns=DROP_COLS, errors="ignore")
    X = X.select_dtypes(include=np.number)
    """
    drop_cols = ["fire_occurred", "fire_cause", "timestamp", "split"]
    X = df_fire.drop(columns=drop_cols, errors="ignore")
    return X.select_dtypes(include=np.number)


# ===================================================
# SIDEBAR – DATA & MODEL STATUS
# ===================================================

with st.sidebar:
    st.header("⚙ Data & Models")

    df = load_dataset()


    if df is None:
        st.error("❌ Dataset not found at:\n`data/features/fire_training_master_clean.csv`")
    else:
        st.success("✅ Dataset loaded")
        st.write(f"Rows: `{df.shape[0]:,}`")
        st.write(f"Columns: `{df.shape[1]}`")

        if "fire_occurred" in df.columns:
            pos = int((df["fire_occurred"] == 1).sum())
            neg = int((df["fire_occurred"] == 0).sum())
            st.caption(f"`fire_occurred` → 1: {pos:,} | 0: {neg:,}")

        if "fire_cause" in df.columns:
            st.caption("`fire_cause` value counts:")
            st.text(df["fire_cause"].value_counts().to_string())

    det_model = load_model(DETECTION_MODEL_FILE)
    cause_model = load_model(CAUSE_MODEL_FILE) if LIGHTGBM_AVAILABLE else None
    spread_model = load_model(SPREAD_MODEL_FILE) if LIGHTGBM_AVAILABLE else None

    st.markdown("### 🧠 Models")
    st.write("Detection:", "✅ Loaded" if det_model is not None else "❌ Missing")
    st.write("Root Cause:", "✅ Loaded" if cause_model is not None else ("⚠ Not loaded (LightGBM missing)" if not LIGHTGBM_AVAILABLE else "❌ Missing file"))
    st.write("Spread:", "✅ Loaded" if spread_model is not None else ("⚠ Not loaded (LightGBM missing)" if not LIGHTGBM_AVAILABLE else "❌ Missing file"))

    if not LIGHTGBM_AVAILABLE:
        st.warning("`lightgbm` is not installed. Cause/Spread models cannot be unpickled.\nRun: `pip install lightgbm` in your venv.")


# ===================================================
# TABS
# ===================================================

tab_overview, tab_detection, tab_shap, tab_temporal, tab_cause_spread, tab_map = st.tabs(
    [
        "1️⃣ Overview",
        "2️⃣ Detection Metrics",
        "3️⃣ SHAP Explainability",
        "4️⃣ Temporal + Ensemble",
        "5️⃣ Root Cause & Spread",
        "6️⃣ Risk Map",
    ]
)


# ===================================================
# TAB 1 – OVERVIEW
# ===================================================

with tab_overview:
    st.subheader("1️⃣ High-Level Overview")

    col1, col2, col3 = st.columns(3)

    # Detection summary from JSON
    summary = load_json_safe(SUMMARY_JSON)

    with col1:
        st.markdown("### 🔍 Detection")
        if summary:
            st.metric("ROC-AUC", f"{summary.get('roc_auc', 0):.3f}")
            st.metric("PR-AUC", f"{summary.get('pr_auc', 0):.3f}")
            st.metric("Test Samples", f"{summary.get('n_test', 0):,}")
        else:
            st.info("Detection summary JSON not found.")

    with col2:
        st.markdown("### 🧠 Root Cause")
        if df is not None and cause_model is not None and "fire_cause" in df.columns:
            st.caption("Evaluated on available labelled records in tab 5.")
            # We will compute metrics in tab 5
            st.metric("Status", "Ready")
        else:
            st.info("Model or target not available.")

    with col3:
        st.markdown("### 🌡️ Spread Severity")
        if df is not None and spread_model is not None and "spread_ha" in df.columns and "fire_occurred" in df.columns:
            st.caption("Evaluated only on `fire_occurred = 1` in tab 5.")
            st.metric("Status", "Ready")
        else:
            st.info("Model or target not available.")

    st.markdown("---")
    st.markdown("### 🧮 Cross-Task Model Comparison (Detection vs Spread)")

    model_comp = load_csv_safe(MODEL_COMPARISON_CSV)
    if model_comp is None:
        st.info("`model_comparison_summary.csv` not found.")
    else:
        st.dataframe(model_comp.round(4), width="stretch")

        metric_cols = pick_metric_columns(
            model_comp,
            ["roc_auc", "pr_auc", "f1_score", "accuracy", "r2", "rmse", "mae"],
        )

        if metric_cols:
            selected_metric = st.selectbox("Metric to compare", metric_cols, key="metric_overview")
            x_col = "task" if "task" in model_comp.columns else metric_cols[0]
            color_col = "type" if "type" in model_comp.columns else None

            fig = px.bar(
                model_comp,
                x=x_col,
                y=selected_metric,
                color=color_col,
                text_auto=".3f",
                title=f"Model Comparison – {selected_metric}",
            )
            fig.update_layout(xaxis_title=x_col.title(), yaxis_title=selected_metric.upper())
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No usable metric columns to plot.")


# ===================================================
# TAB 2 – DETECTION METRICS
# ===================================================

with tab_detection:
    st.subheader("2️⃣ Detection – Thresholds & Curves")

    cm = load_confusion_matrix()
    thr_df = load_csv_safe(THRESHOLD_SWEEP_CSV)
    roc_df = load_csv_safe(ROC_CSV)
    pr_df = load_csv_safe(PR_CSV)
    scenarios = load_json_safe(SCENARIOS_JSON)

    col1, col2 = st.columns(2)

    # ---- Confusion matrix + derived metrics
    with col1:
        st.markdown("### 🧾 Confusion Matrix (Threshold = 0.25)")
        if cm is None:
            st.warning("Confusion matrix text file not found or malformed.")
        else:
            tn, fp, fn, tp = cm.ravel()
            cm_df = pd.DataFrame(
                cm,
                columns=["Pred: No Fire", "Pred: Fire"],
                index=["Actual: No Fire", "Actual: Fire"],
            )
            st.dataframe(cm_df, width="stretch")

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            acc = (tp + tn) / cm.sum() if cm.sum() else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{precision:.3f}")
            c3.metric("Recall", f"{recall:.3f}")
            c4.metric("F1", f"{f1:.3f}")

    # ---- Threshold explorer
    with col2:
        st.markdown("### 🎚 Threshold Explorer")
        if thr_df is not None and "threshold" in thr_df.columns:
            t_min = float(thr_df["threshold"].min())
            t_max = float(thr_df["threshold"].max())
            default_t = 0.25 if t_min <= 0.25 <= t_max else float(thr_df["threshold"].iloc[0])

            t = st.slider(
                "Decision threshold (probability cutoff for FIRE)",
                min_value=float(t_min),
                max_value=float(t_max),
                value=float(default_t),
                step=0.01,
                key="thr_detection",
            )

            idx = (thr_df["threshold"] - t).abs().idxmin()
            row_thr = thr_df.loc[idx]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{row_thr.get('precision', 0):.3f}")
            c2.metric("Recall", f"{row_thr.get('recall', 0):.3f}")
            c3.metric("F1-score", f"{row_thr.get('f1', 0):.3f}")
            c4.metric("False Alarm Rate", f"{row_thr.get('false_alarm_rate', 0):.3f}")
        else:
            st.info("Threshold sweep CSV not found or missing 'threshold' column.")

    st.markdown("#### Full Threshold Sweep Table")
    if thr_df is not None:
        st.dataframe(thr_df.round(4), width="stretch")

    st.markdown("---")
    col3, col4 = st.columns(2)

    # ---- ROC curve
    with col3:
        st.markdown("### ROC Curve")
        if roc_df is not None:
            fpr_col = get_first_matching_column(roc_df, ["fpr", "false_positive"])
            tpr_col = get_first_matching_column(roc_df, ["tpr", "true_positive"])
            if fpr_col and tpr_col:
                fig_roc = px.line(
                    roc_df,
                    x=fpr_col,
                    y=tpr_col,
                    title="ROC Curve – Detection Model",
                )
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, width="stretch")
            else:
                st.warning("Could not detect FPR/TPR columns.")
        else:
            st.info("ROC CSV not found.")

    # ---- PR curve
    with col4:
        st.markdown("### Precision–Recall Curve")
        if pr_df is not None:
            prec_col = get_first_matching_column(pr_df, ["precision", "prec"])
            rec_col = get_first_matching_column(pr_df, ["recall"])
            if prec_col and rec_col:
                fig_pr = px.line(
                    pr_df,
                    x=rec_col,
                    y=prec_col,
                    title="Precision–Recall Curve – Detection Model",
                )
                fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig_pr, width="stretch")
            else:
                st.warning("Could not detect precision/recall columns.")
        else:
            st.info("PR CSV not found.")

    st.markdown("---")
    st.markdown("### 🧩 Early-Warning Scenarios")
    if scenarios:
        st.json(scenarios)
    else:
        st.info("`detection_threshold_scenarios.json` not found or invalid.")


# ===================================================
# TAB 3 – SHAP
# ===================================================

with tab_shap:
    st.subheader("3️⃣ SHAP Explainability – What drives risk?")

    shap_imp = load_csv_safe(SHAP_IMPORTANCE_CSV)

    st.markdown("### 📈 Global Feature Importance")
    if shap_imp is not None and not shap_imp.empty:
        fcol = get_first_matching_column(shap_imp, ["feature", "name"])
        vcol = get_first_matching_column(shap_imp, ["shap", "importance", "value"])

        if fcol and vcol:
            top_k = shap_imp.sort_values(vcol, ascending=False).head(20)
            fig = px.bar(
                top_k,
                x=vcol,
                y=fcol,
                orientation="h",
                title="Top SHAP Feature Contributions (Mean |SHAP|)",
            )
            fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Could not infer feature/importance columns. Showing raw table.")
            st.dataframe(shap_imp, width="stretch")
    else:
        st.info("SHAP importance CSV not found or empty.")

    st.markdown("### 🖼 SHAP Summary Plot")
    if SHAP_SUMMARY_PNG.exists():
        st.image(str(SHAP_SUMMARY_PNG), width="stretch")
    else:
        st.info("`shap_summary_bar.png` not found.")

    st.markdown("---")
    st.markdown("### 🔎 Local Explanations (Sample Records)")
    shap_local = load_csv_safe(SHAP_TOP_RECORDS_CSV)
    if shap_local is not None and not shap_local.empty:
        st.dataframe(shap_local.head(50), width="stretch")
    else:
        st.info("`shap_top_records.csv` not found or empty.")


# ===================================================
# TAB 4 – TEMPORAL + ENSEMBLE
# ===================================================

with tab_temporal:
    st.subheader("4️⃣ Temporal CV & Ensemble Rules")

    temp_df = load_csv_safe(TEMPORAL_CV_CSV)
    if temp_df is not None and not temp_df.empty:
        year_col = "year" if "year" in temp_df.columns else temp_df.columns[0]
        value_cols = [c for c in ["roc_auc", "pr_auc"] if c in temp_df.columns]

        if value_cols:
            melted = temp_df.melt(
                id_vars=year_col,
                value_vars=value_cols,
                var_name="metric",
                value_name="value",
            )
            fig = px.line(
                melted,
                x=year_col,
                y="value",
                color="metric",
                markers=True,
                title="Temporal Cross-Validation – Performance by Year",
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="Score")
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Raw Temporal CV Table")
        st.dataframe(temp_df.round(4), width="stretch")
    else:
        st.info("Temporal CV CSV not found or empty.")

    st.markdown("---")
    st.markdown("### 🧠 Ensemble Decision Rules (v2)")

    ens_df = load_csv_safe(ENSEMBLE_V2_CSV)
    if ens_df is not None and not ens_df.empty:
        st.dataframe(ens_df.round(4), width="stretch")

        for idx, r in ens_df.iterrows():
            label = safe_rule_label(r, ens_df, idx)
            st.markdown(f"**Rule:** `{label}`")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{r.get('accuracy', 0):.3f}")
            c2.metric("Precision", f"{r.get('precision', 0):.3f}")
            c3.metric("Recall", f"{r.get('recall', 0):.3f}")
            c4.metric("F1", f"{r.get('f1', 0):.3f}")

            tp = int(r.get("tp", 0)) if "tp" in ens_df.columns else 0
            fp = int(r.get("fp", 0)) if "fp" in ens_df.columns else 0
            tn = int(r.get("tn", 0)) if "tn" in ens_df.columns else 0
            fn = int(r.get("fn", 0)) if "fn" in ens_df.columns else 0
            st.caption(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            st.divider()
    else:
        st.info("`ensemble_rules_comparison_v2.csv` not found or empty.")


# ===================================================
# TAB 5 – ROOT CAUSE & SPREAD PERFORMANCE
# ===================================================

with tab_cause_spread:
    st.subheader("5️⃣ Root Cause & Spread – Live Evaluation from Dataset")

    if df is None:
        st.error("Dataset not loaded – cannot evaluate models.")
    else:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            accuracy_score,
            f1_score,
            mean_squared_error,
            mean_absolute_error,
            r2_score,
        )

        # ---------- ROOT CAUSE ----------
        st.markdown("### 🧠 Root Cause Classification (Human / Lightning / Unknown)")

        if cause_model is not None and "fire_cause" in df.columns:
            df_c = df[df["fire_cause"].notna()].copy()
            if "split" in df_c.columns and "test" in df_c["split"].unique():
                df_c = df_c[df_c["split"] == "test"].copy()

            if len(df_c) > 0:
                X_cause = build_features_cause(df_c)
                y_true_c = df_c["fire_cause"]
                y_pred_c = cause_model.predict(X_cause)

                acc = accuracy_score(y_true_c, y_pred_c)
                macro_f1 = f1_score(y_true_c, y_pred_c, average="macro")
                weighted_f1 = f1_score(y_true_c, y_pred_c, average="weighted")

                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Macro F1", f"{macro_f1:.3f}")
                c3.metric("Weighted F1", f"{weighted_f1:.3f}")

                st.markdown("#### Per-Class Report")
                report_dict = classification_report(
                    y_true_c,
                    y_pred_c,
                    output_dict=True,
                    zero_division=0,
                )
                report_df = pd.DataFrame(report_dict).T
                st.dataframe(report_df.round(3), width="stretch")

                st.markdown("#### Confusion Matrix")
                cm_c = confusion_matrix(y_true_c, y_pred_c, labels=sorted(y_true_c.unique()))
                cm_c_df = pd.DataFrame(
                    cm_c,
                    index=[f"Actual: {c}" for c in sorted(y_true_c.unique())],
                    columns=[f"Pred: {c}" for c in sorted(y_true_c.unique())],
                )
                st.dataframe(cm_c_df, width="stretch")
            else:
                st.info("No labelled `fire_cause` rows available for evaluation.")
        else:
            st.info("Root cause model or target column not available.")

        st.markdown("---")

        # ---------- SPREAD REGRESSION ----------
        st.markdown("### 🌡️ Fire Spread Regression (hectares)")

        if spread_model is not None and "spread_ha" in df.columns and "fire_occurred" in df.columns:
            df_s = df[(df["fire_occurred"] == 1) & df["spread_ha"].notna()].copy()
            if "split" in df_s.columns and "test" in df_s["split"].unique():
                df_s = df_s[df_s["split"] == "test"].copy()

            if len(df_s) > 0:
                X_spread = build_features_spread(df_s)
                y_true = df_s["spread_ha"].clip(lower=0)

                # Trained on log1p(spread_ha)
                preds_log = spread_model.predict(X_spread)
                y_pred = np.expm1(preds_log)

                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE (ha)", f"{rmse:.2f}")
                c2.metric("MAE (ha)", f"{mae:.2f}")
                c3.metric("R²", f"{r2:.3f}")

                st.markdown("#### Sample Predictions (First 50)")
                sample_df = df_s[["spread_ha"]].copy()
                sample_df["pred_spread_ha"] = y_pred
                st.dataframe(sample_df.head(50).round(2), width="stretch")
            else:
                st.info("No `fire_occurred=1` rows with `spread_ha` available for evaluation.")
        else:
            st.info("Spread model or required columns not available.")
# =====================================================================
# DETECT REAL GPS COORDINATES (or fallback safely)
# =====================================================================

def get_map_coords(df):

    # Preferred real columns
    possible_lat = ["latitude", "lat_real", "lat_raw"]
    possible_lon = ["longitude", "lon_real", "lon_raw"]

    for lat in possible_lat:
        for lon in possible_lon:
            if lat in df.columns and lon in df.columns:
                return lat, lon, False

    # Fallback: scaled columns
    if "lat" in df.columns and "lon" in df.columns:
        return "lat", "lon", True

    raise ValueError("No coordinate columns found!")


lat_col, lon_col, is_scaled = get_map_coords(df)

if is_scaled:
    st.warning("""
⚠️ Geographic data is **scaled for ML training**, not real GPS coordinates.

Map points will appear near Africa/Atlantic instead of Canada.
To enable true mapping:
add real latitude/longitude columns before scaling.
""")


# ===================================================
# TAB 6 – RISK MAP
# ===================================================

with tab_map:
    st.subheader("6️⃣ Risk Map – Sample Points with Detection Probability")

    if df is None:
        st.error("Dataset not loaded – cannot render map.")
    elif det_model is None:
        st.error("Detection model not loaded – cannot compute risk scores.")
    elif not FOLIUM_AVAILABLE:
        st.warning("`folium` or `streamlit_folium` not installed. Run:\n`pip install folium streamlit-folium`")
    else:
        # Build detection features to get probabilities
        X_det = build_features_detection(df)
        try:
            probs = det_model.predict_proba(X_det)[:, 1]
        except Exception as e:
            st.error(f"Could not compute detection probabilities: {e}")
        else:
            df_map = df.copy()
            df_map["det_prob"] = probs

            # Try to get lat/lon for map
            # If you have REAL lat/lon columns (not standardized), use them directly.
            lat_col, lon_col, is_scaled = get_map_coords(df)


            if lat_col not in df_map.columns or lon_col not in df_map.columns:
                st.error("No `lat` / `lon` columns found for map.")
            else:
                # Sample subset for performance
                N = st.slider("Number of points to plot", 200, 2000, 800, step=100, key="n_map")
                df_sample = df_map.sample(n=min(N, len(df_map)), random_state=42)

                # Basic risk categories from probability
                def risk_category(p):
                    if p >= 0.8:
                        return "Extreme"
                    elif p >= 0.6:
                        return "High"
                    elif p >= 0.4:
                        return "Medium"
                    else:
                        return "Low"

                df_sample["risk_category"] = df_sample["det_prob"].apply(risk_category)

                # Center map around median lat/lon (even if standardized, this is just for demo)
                center_lat = float(df_sample[lat_col].median())
                center_lon = float(df_sample[lon_col].median())

                m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

                for _, row in df_sample.iterrows():
                    p = float(row["det_prob"])
                    risk = row["risk_category"]

                    if risk == "Extreme":
                        color = "red"
                    elif risk == "High":
                        color = "orange"
                    elif risk == "Medium":
                        color = "yellow"
                    else:
                        color = "green"

                    popup_items = [
                        f"Detection prob: {p:.3f}",
                    ]
                    if "fire_occurred" in row.index:
                        popup_items.append(f"fire_occurred: {int(row['fire_occurred'])}")
                    if "fire_cause" in row.index:
                        popup_items.append(f"fire_cause: {row['fire_cause']}")
                    if "spread_ha" in row.index:
                        popup_items.append(f"spread_ha: {row['spread_ha']}")

                    popup_text = "<br>".join(popup_items)

                    try:
                        folium.CircleMarker(
                            location=[float(row[lat_col]), float(row[lon_col])],
                            radius=4,
                            color=color,
                            fill=True,
                            fill_opacity=0.7,
                            popup=folium.Popup(popup_text, max_width=300),
                        ).add_to(m)
                    except Exception:
                        continue

                st_folium(m, width=900, height=550)
