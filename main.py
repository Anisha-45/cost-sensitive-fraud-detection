# =============================================================================
# main.py — Credit Card Fraud Detection  |  Full Pipeline Orchestrator
# =============================================================================
# Run:  python main.py
#
# Expects fraudTrain.csv and fraudTest.csv in the same folder (or update
# TRAIN_PATH / TEST_PATH below).
#
# Outputs (all saved to ./outputs/):
#   eda_overview.png          — 6-panel EDA figure
#   model_comparison.png      — 4-metric bar chart across models
#   threshold_analysis.png    — cost & metric curves vs threshold
#   confusion_matrices.png    — annotated confusion matrices
#   roc_curves.png            — ROC curves for all models
#   pr_curves.png             — Precision-Recall curves
#   feature_importance.png    — top feature importance chart
#   before_after_comparison.png — optimisation impact chart
#   best_model.pkl            — serialised model bundle
# =============================================================================

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
sys.path.insert(0, ".")

# ── Module imports ─────────────────────────────────────────────────────────────
from module1_data_pipeline import (
    load_data,
    engineer_features,
    run_eda,
    preprocess,
    FEATURE_COLS,
)
from module2_model_training import (
    train_all_models,
    plot_model_comparison,
    plot_threshold_analysis,
)
from module3_evaluation import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_pr_curves,
    plot_feature_importance,
    plot_before_after,
    print_final_report,
)

# =============================================================================
# CONFIGURATION  — update paths if your CSVs are elsewhere
# =============================================================================
TRAIN_PATH = "fraudTrain.csv"   # or full path
TEST_PATH  = "fraudTest.csv"

APPLY_SMOTE    = True
SMOTE_STRATEGY = 0.20    # target minority/majority ratio after SMOTE
RUN_TUNING     = True    # set False for a quicker run (skips RandomizedSearchCV)
OUTPUT_DIR     = "outputs"


# =============================================================================
def banner(text: str, width: int = 62):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


# =============================================================================
def main():
    total_start = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    banner("MODULE 1 — Data Pipeline & Preprocessing")
    # ──────────────────────────────────────────────────────────────────────────

    # 1a. Load raw CSVs
    df_train_raw, df_test_raw = load_data(TRAIN_PATH, TEST_PATH)

    # 1b. Feature engineering (temporal, geo, amount transforms, encoding)
    print("\n  Engineering features …")
    df_train = engineer_features(df_train_raw)
    df_test  = engineer_features(df_test_raw)

    # 1c. Exploratory data analysis
    run_eda(df_train, save_path=os.path.join(OUTPUT_DIR, "eda_overview.png"))

    # 1d. Preprocess: scale + SMOTE on training set only
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df_train, df_test,
        apply_smote    = APPLY_SMOTE,
        smote_strategy = SMOTE_STRATEGY,
    )

    # Keep test amounts for cost calculations
    test_amounts = df_test["amt"].values

    # ──────────────────────────────────────────────────────────────────────────
    banner("MODULE 2 — Model Training & Optimisation")
    # ──────────────────────────────────────────────────────────────────────────

    # 2a. Train all models, tune best two, optimise threshold
    results, best_name, best_model, opt_t, thr_data = train_all_models(
        X_train, X_test, y_train, y_test,
        feature_names = FEATURE_COLS,
        test_amounts  = test_amounts,
        run_tuning    = RUN_TUNING,
        output_dir    = OUTPUT_DIR,
    )

    # 2b. Visualisations from module 2
    plot_model_comparison(
        results,
        save_path = os.path.join(OUTPUT_DIR, "model_comparison.png"),
    )
    plot_threshold_analysis(
        thr_data,
        save_path = os.path.join(OUTPUT_DIR, "threshold_analysis.png"),
    )

    # ──────────────────────────────────────────────────────────────────────────
    banner("MODULE 3 — Evaluation & Visualisations")
    # ──────────────────────────────────────────────────────────────────────────

    # 3a. Confusion matrices with dollar annotations
    plot_confusion_matrices(
        results, y_test, test_amounts,
        save_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
    )

    # 3b. ROC curves
    plot_roc_curves(
        results, y_test,
        save_path = os.path.join(OUTPUT_DIR, "roc_curves.png"),
    )

    # 3c. Precision-Recall curves
    plot_pr_curves(
        results, y_test,
        save_path = os.path.join(OUTPUT_DIR, "pr_curves.png"),
    )

    # 3d. Feature importance
    plot_feature_importance(
        best_model, FEATURE_COLS,
        save_path = os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )

    # 3e. Before / after threshold optimisation comparison
    plot_before_after(
        results, y_test,
        save_path = os.path.join(OUTPUT_DIR, "before_after_comparison.png"),
    )

    # 3f. Printed summary table
    print_final_report(results)

    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    banner(f"Pipeline complete  —  {elapsed:.1f} s")

    print("\n  Output files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath) // 1024
        print(f"    {fname:<40}  {size:>6} KB")
    print()


# =============================================================================
if __name__ == "__main__":
    main()
