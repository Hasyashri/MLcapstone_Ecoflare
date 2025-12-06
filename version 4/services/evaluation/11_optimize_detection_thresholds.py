"""
STEP 11 — DETECTION THRESHOLD OPTIMIZATION (VISUAL STRATEGY)

Consumes outputs from Step 10:

- threshold_sweep_detection.csv
- early_warning_thresholds_detection.csv
- summary_metrics_detection.json

Generates:

- detection_threshold_scenarios.csv
- detection_threshold_scenarios.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# ------------------------------------------------------
# FIX IMPORT PATH FOR LOGGER (Windows-safe)
# ------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVICES_DIR = os.path.join(PROJECT_ROOT, "services")

if SERVICES_DIR not in sys.path:
    sys.path.insert(0, SERVICES_DIR)

from training.logger import get_logger  # ✅ NOW WORKS


log = get_logger("STEP_11_DETECTION_THRESHOLD_OPT")

# ------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------

EVAL_DIR = os.path.join(PROJECT_ROOT, "reports", "detection_eval")

THR_SWEEP_FILE = os.path.join(EVAL_DIR, "threshold_sweep_detection.csv")
EARLY_WARN_FILE = os.path.join(EVAL_DIR, "early_warning_thresholds_detection.csv")
SUMMARY_FILE = os.path.join(EVAL_DIR, "summary_metrics_detection.json")

SCENARIOS_CSV = os.path.join(EVAL_DIR, "detection_threshold_scenarios.csv")
SCENARIOS_JSON = os.path.join(EVAL_DIR, "detection_threshold_scenarios.json")


# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------

REQ_COLS = {"threshold", "precision", "recall", "f1", "false_alarm_rate"}


def load_threshold_sweep():
    if not os.path.exists(THR_SWEEP_FILE):
        raise FileNotFoundError("threshold_sweep_detection.csv not found. Run STEP 10 first.")

    df = pd.read_csv(THR_SWEEP_FILE)

    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in threshold_sweep file: {missing}")

    return df


def load_early_warning():
    if not os.path.exists(EARLY_WARN_FILE):
        log.warning("Early warning file not found — proceeding without it.")
        return None

    df = pd.read_csv(EARLY_WARN_FILE, index_col=0)
    return df


def load_summary():
    if not os.path.exists(SUMMARY_FILE):
        log.warning("Summary metrics JSON not found — proceeding without it.")
        return {}

    with open(SUMMARY_FILE, "r") as f:
        return json.load(f)


def make_scenario_row(name, row, summary):
    return {
        "scenario_name": name,
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "false_alarm_rate": float(row["false_alarm_rate"]),
        "roc_auc": float(summary.get("roc_auc", np.nan)),
        "pr_auc": float(summary.get("pr_auc", np.nan))
    }


def attach_descriptions(df):
    descriptions = []

    for _, r in df.iterrows():
        name = r["scenario_name"]
        desc = (
            f"Threshold={r['threshold']:.2f} | "
            f"Recall={r['recall']:.2f}, "
            f"Precision={r['precision']:.2f}, "
            f"False Alarms={r['false_alarm_rate']:.2f}"
        )

        if name == "BALANCED":
            desc = "Best overall F1 score. " + desc
        elif name == "LOW_FALSE_ALARM":
            desc = "Low false-positive operating point. " + desc
        elif name.startswith("AGGRESSIVE_EARLY"):
            desc = "High-recall early-warning strategy. " + desc

        descriptions.append(desc)

    df["description"] = descriptions
    return df


def build_extra_candidates(thr_df, used_thresholds):
    results = {}

    # AGGRESSIVE EARLY WARNING:
    aggressive = thr_df[thr_df["precision"] >= 0.05]
    if aggressive.empty:
        aggressive = thr_df.copy()

    pick = aggressive.loc[aggressive["recall"].idxmax()]
    if pick["threshold"] not in used_thresholds:
        results["AGGRESSIVE_EARLY_WARNING"] = pick

    # TOP F1 SCENARIOS
    f1_top = thr_df.sort_values("f1", ascending=False).head(5)
    for _, row in f1_top.iterrows():
        t = row["threshold"]
        if t in used_thresholds:
            continue

        results[f"TOP_F1_{str(t).replace('.', '_')}"] = row

        if len(results) >= 4:
            break

    return results


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

def main():
    log.info("===== DETECTION THRESHOLD OPTIMIZATION STARTED =====")

    thr_df = load_threshold_sweep()
    early_df = load_early_warning()
    summary = load_summary()

    scenarios = {}
    used_thresholds = set()

    # EARLY WARNING SCENARIOS
    if early_df is not None and not early_df.empty:
        log.info("Using thresholds from early-warning file.")

        for name, row in early_df.iterrows():
            t = float(row["threshold"])
            used_thresholds.add(t)
            scenarios[name] = make_scenario_row(name, row, summary)

    # ADD EXTRA CANDIDATES
    extras = build_extra_candidates(thr_df, used_thresholds)
    for name, row in extras.items():
        scenarios[name] = make_scenario_row(name, row, summary)

    if not scenarios:
        raise RuntimeError("No scenarios could be constructed.")

    df = pd.DataFrame(list(scenarios.values()))
    df = attach_descriptions(df)
    df = df.sort_values("f1", ascending=False)

    # SAVE
    df.to_csv(SCENARIOS_CSV, index=False)
    df.to_json(SCENARIOS_JSON, orient="records", indent=2)

    log.info(f"Saved CSV  -> {SCENARIOS_CSV}")
    log.info(f"Saved JSON -> {SCENARIOS_JSON}")


    
    log.info("===== DETECTION THRESHOLD OPTIMIZATION COMPLETE =====")


if __name__ == "__main__":
    main()
