# =============================================================================
# MODULE 3 — Evaluation, Visualisations & Feature Importance
# =============================================================================
# Responsibilities:
#   • Annotated confusion matrices (dollar amounts)
#   • ROC curves for all models
#   • Precision-Recall curves for all models
#   • Feature importance plot (bar chart, colour-coded by rank)
#   • Before / after optimisation comparison (metrics + cost)
#   • Printed final performance report table
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ── Colours ────────────────────────────────────────────────────────────────────
BG = "#0f1117"; PANEL = "#1a1d2e"; BORDER = "#3a3d52"; TEXT = "#c9d1d9"
MUTED = "#8b949e"; RED = "#ff4757"; GREEN = "#2ed573"; ORANGE = "#ffa502"
BLUE = "#1e90ff"; PURPLE = "#a29bfe"

plt.rcParams.update({
    "figure.facecolor": BG,    "axes.facecolor":   PANEL,
    "axes.edgecolor":   BORDER,"axes.labelcolor":  TEXT,
    "xtick.color":      MUTED, "ytick.color":      MUTED,
    "text.color":       TEXT,  "grid.color":       BORDER,
    "grid.linestyle":   "--",  "grid.alpha":       0.4,
    "legend.facecolor": PANEL, "legend.edgecolor": BORDER,
})

DARK_CM = LinearSegmentedColormap.from_list(
    "fraud_heatmap", [PANEL, ORANGE, RED], N=256
)

MODEL_COLORS = [BLUE, GREEN, ORANGE, RED, PURPLE]


# =============================================================================
# 1. CONFUSION MATRICES
# =============================================================================
def plot_confusion_matrices(results: dict, y_test: np.ndarray,
                             test_amounts: np.ndarray,
                             save_path: str = "outputs/confusion_matrices.png"):
    """
    One confusion-matrix panel per model (up to 4).
    Each cell shows:  count on top  |  dollar annotation below.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Limit to 4 models to keep the figure readable
    items = list(results.items())[:4]
    n     = len(items)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)

    class_labels = ["Legit", "Fraud"]
    mean_fraud   = float(test_amounts[y_test == 1].mean()) \
                   if (y_test == 1).any() else 200.0

    for ax, (name, res) in zip(axes, items):
        threshold = res.get("threshold", 0.50)
        preds     = (res["probs"] >= threshold).astype(int)
        cm        = confusion_matrix(y_test, preds)

        sns.heatmap(
            cm, annot=False, fmt="d", cmap=DARK_CM, ax=ax,
            linewidths=2, linecolor=BG, cbar=False,
            xticklabels=class_labels, yticklabels=class_labels,
        )

        # Manual annotations: count + dollar impact
        cell_info = {
            (0, 0): (GREEN,  "✓ Correct"),
            (0, 1): (ORANGE, f"~${cm[0,1]*10:,.0f}\nreview cost"),
            (1, 0): (RED,    f"~${cm[1,0]*mean_fraud:,.0f}\nlost to fraud"),
            (1, 1): (GREEN,  f"~${cm[1,1]*mean_fraud:,.0f}\nsaved"),
        }
        for (row, col), (color, note) in cell_info.items():
            ax.text(col + 0.5, row + 0.33, f"{cm[row, col]:,}",
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white")
            ax.text(col + 0.5, row + 0.72, note,
                    ha="center", va="center",
                    fontsize=8.5, color=color)

        t_str = f"t = {threshold:.2f}"
        ax.set_title(f"{name}\n({t_str})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — Financial Impact View",
                 fontsize=15, fontweight="bold", y=1.03, color="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Confusion matrices saved  →  {save_path}")


# =============================================================================
# 2. ROC CURVES
# =============================================================================
def plot_roc_curves(results: dict, y_test: np.ndarray,
                    save_path: str = "outputs/roc_curves.png"):
    """
    ROC curves for all models on a single axis.
    Shaded area under the best-model curve.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Exclude optimised duplicate (same probs, just different threshold)
    base_results = {k: v for k, v in results.items() if "Optimised" not in k}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    for (name, res), color in zip(base_results.items(), MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        auc         = res["roc_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{name}  (AUC = {auc:.4f})")

    # Shade under the first model's curve as a visual guide
    first_res = next(iter(base_results.values()))
    fpr0, tpr0, _ = roc_curve(y_test, first_res["probs"])
    ax.fill_between(fpr0, tpr0, alpha=0.07, color=MODEL_COLORS[0])

    ax.plot([0, 1], [0, 1], color=BORDER, linestyle="--",
            linewidth=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ROC curves saved          →  {save_path}")


# =============================================================================
# 3. PRECISION-RECALL CURVES
# =============================================================================
def plot_pr_curves(results: dict, y_test: np.ndarray,
                   save_path: str = "outputs/pr_curves.png"):
    """
    Precision-Recall curves.
    Preferred metric over ROC-AUC when classes are highly imbalanced.
    A high AUC-PR means the model is both catching fraud (recall) and
    keeping false alarms low (precision).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    base_results = {k: v for k, v in results.items() if "Optimised" not in k}
    baseline     = float(y_test.mean())

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    for (name, res), color in zip(base_results.items(), MODEL_COLORS):
        prec, rec, _ = precision_recall_curve(y_test, res["probs"])
        ap           = res["auc_pr"]
        ax.plot(rec, prec, color=color, linewidth=2.5,
                label=f"{name}  (AP = {ap:.4f})")

    ax.axhline(baseline, color=BORDER, linestyle="--", linewidth=1.5,
               label=f"No-Skill baseline ({baseline:.3f})")

    ax.set_xlabel("Recall (Fraud Caught Rate)", fontsize=12)
    ax.set_ylabel("Precision (Correct Fraud Flags)", fontsize=12)
    ax.set_title("Precision-Recall Curves\n"
                 "(Better metric than ROC-AUC for imbalanced data)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  PR curves saved           →  {save_path}")


# =============================================================================
# 4. FEATURE IMPORTANCE
# =============================================================================
def plot_feature_importance(model, feature_names: list, top_n: int = 17,
                             save_path: str = "outputs/feature_importance.png"):
    """
    Horizontal bar chart of the top-N features.
    Works for any model that exposes feature_importances_ or coef_.
    Bars are colour-coded: red = top-5, orange = 6-10, blue = rest.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("  ⚠  Model does not expose feature importances — skipping.")
        return

    fi = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(top_n))

    colors = [RED if i < 5 else ORANGE if i < 10 else BLUE
              for i in range(len(fi))]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    bars = ax.barh(fi["Feature"][::-1], fi["Importance"][::-1],
                   color=colors[::-1], alpha=0.85, edgecolor=BG)
    for bar, val in zip(bars, fi["Importance"][::-1]):
        ax.text(val + fi["Importance"].max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=9, color=TEXT)

    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top {top_n} Features for Fraud Detection",
                 fontsize=14, fontweight="bold")
    ax.legend(
        handles=[Patch(facecolor=RED,    label="Top 5 (Critical)"),
                 Patch(facecolor=ORANGE, label="Rank 6–10 (Important)"),
                 Patch(facecolor=BLUE,   label="Rank 11+ (Useful)")],
        fontsize=10, framealpha=0.3,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Feature importance saved  →  {save_path}")


# =============================================================================
# 5. BEFORE / AFTER OPTIMISATION
# =============================================================================
def plot_before_after(results: dict, y_test: np.ndarray,
                       save_path: str = "outputs/before_after_comparison.png"):
    """
    Side-by-side grouped bar charts:
    Left  : Recall / Precision / F1 / ROC-AUC before vs after threshold tuning.
    Right : FP cost / FN cost / Total cost / Money saved before vs after.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Find the base model and its optimised counterpart
    opt_key  = next((k for k in results if "Optimised" in k), None)
    if opt_key is None:
        print("  ⚠  No optimised result found — skipping before/after plot.")
        return
    base_key = opt_key.replace(" (Optimised)", "")

    perf_metrics = ["recall", "precision", "f1", "roc_auc"]
    perf_labels  = ["Recall", "Precision", "F1", "ROC-AUC"]
    before_perf  = [results[base_key][m] for m in perf_metrics]
    after_perf   = [results[opt_key][m]  for m in perf_metrics]

    cost_keys   = ["FP_cost", "FN_cost", "Total_cost", "Money_saved"]
    cost_labels = ["FP Cost ($)", "FN Cost ($)", "Total Cost ($)", "Saved ($)"]
    before_cost = [results[base_key]["cost"][k] for k in cost_keys]
    after_cost  = [results[opt_key]["cost"][k]  for k in cost_keys]

    x     = np.arange(4)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG)

    # ── Metrics ────────────────────────────────────────────────────────────────
    ax1.set_facecolor(PANEL)
    b1 = ax1.bar(x - width/2, before_perf, width,
                 label=f"Default (t=0.50)", color=BLUE, alpha=0.75, edgecolor=BG)
    opt_t = results[opt_key].get("threshold", 0.50)
    b2 = ax1.bar(x + width/2, after_perf, width,
                 label=f"Optimised (t={opt_t:.2f})", color=GREEN, alpha=0.75, edgecolor=BG)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax1.set_xticks(x); ax1.set_xticklabels(perf_labels)
    ax1.set_ylim(0, 1.18); ax1.set_ylabel("Score")
    ax1.set_title("Performance Metrics: Before vs After", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)

    # ── Costs ──────────────────────────────────────────────────────────────────
    ax2.set_facecolor(PANEL)
    cell_colors = [ORANGE, RED, "#ff6b81", GREEN]
    b3 = ax2.bar(x - width/2, [v/1e6 for v in before_cost], width,
                 label="Default", color=cell_colors, alpha=0.65, edgecolor=BG)
    b4 = ax2.bar(x + width/2, [v/1e6 for v in after_cost],  width,
                 label="Optimised", color=cell_colors, alpha=1.00, edgecolor=BG)
    max_val = max(before_cost + after_cost)
    for bars, vals in [(b3, before_cost), (b4, after_cost)]:
        for bar, raw in zip(bars, vals):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2,
                     h + max_val/1e6 * 0.012,
                     f"${raw/1e6:.2f}M", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax2.set_xticks(x); ax2.set_xticklabels(cost_labels, rotation=10, ha="right")
    ax2.set_ylabel("Amount ($ millions)")
    ax2.set_title("Financial Impact: Before vs After", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.suptitle(f"{base_key} — Threshold Optimisation Impact",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Before/after plot saved   →  {save_path}")


# =============================================================================
# 6. FINAL REPORT (console table)
# =============================================================================
def print_final_report(results: dict):
    """Print a formatted performance and cost summary table."""
    print("\n" + "═" * 90)
    print("  FINAL MODEL PERFORMANCE REPORT")
    print("═" * 90)
    hdr = (f"{'Model':<32} {'Recall':>7} {'Prec':>7} {'F1':>7} "
           f"{'AUC-PR':>8} {'Threshold':>10} {'Total Cost':>12} {'Saved':>12}")
    print(hdr)
    print("─" * 90)
    for name, res in results.items():
        cost = res.get("cost", {})
        print(
            f"{name:<32} "
            f"{res['recall']:>7.4f} "
            f"{res['precision']:>7.4f} "
            f"{res['f1']:>7.4f} "
            f"{res['auc_pr']:>8.4f} "
            f"{res.get('threshold', 0.5):>10.2f} "
            f"${cost.get('Total_cost', 0):>11,.0f} "
            f"${cost.get('Money_saved', 0):>11,.0f}"
        )
    print("═" * 90)


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from module1_data_pipeline import (load_data, engineer_features,
                                        preprocess, FEATURE_COLS)
    from module2_model_training import train_all_models

    TRAIN = "/mnt/user-data/uploads/fraudTrain.csv"
    TEST  = "/mnt/user-data/uploads/fraudTest.csv"

    df_tr_raw, df_te_raw = load_data(TRAIN, TEST)
    df_tr = engineer_features(df_tr_raw)
    df_te = engineer_features(df_te_raw)

    X_train, X_test, y_train, y_test, scaler = preprocess(df_tr, df_te)
    amounts = df_te["amt"].values

    results, best_name, best_model, opt_t, thr_data = train_all_models(
        X_train, X_test, y_train, y_test, FEATURE_COLS, amounts,
        run_tuning=True,
    )

    plot_confusion_matrices(results, y_test, amounts)
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_feature_importance(best_model, FEATURE_COLS)
    plot_before_after(results, y_test)
    print_final_report(results)
    print("\n  Module 3 complete.")
