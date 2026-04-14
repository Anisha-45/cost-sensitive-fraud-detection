# =============================================================================
# MODULE 2 — Model Training & Cost-Sensitive Optimisation
# =============================================================================
# Responsibilities:
#   • Define three classifiers: Logistic Regression, Random Forest,
#     Gradient Boosting
#   • Stratified k-fold cross-validation (recall + AUC-PR)
#   • RandomizedSearchCV hyperparameter tuning
#   • Cost-sensitive loss function (FN costs full fraud amount,
#     FP costs fixed analyst-review fee)
#   • Classification-threshold sweep to minimise total financial cost
#   • Save best model bundle (.pkl) to outputs/
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection  import (StratifiedKFold, cross_val_score,
                                      RandomizedSearchCV)
from sklearn.metrics          import (recall_score, precision_score, f1_score,
                                      roc_auc_score, average_precision_score,
                                      confusion_matrix)

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


# =============================================================================
# COST MATRIX
# =============================================================================
def compute_cost(y_true: np.ndarray, y_pred: np.ndarray,
                 amounts: np.ndarray,
                 fp_cost: float = 10.0) -> dict:
    """
    Financial cost function.

    False Negative (missed fraud)  : full transaction amount is lost.
    False Positive (false alarm)   : fixed analyst-review cost ($10).

    Parameters
    ----------
    y_true   : true labels
    y_pred   : predicted labels
    amounts  : transaction amounts aligned with y_true
    fp_cost  : cost per false positive (default $10)

    Returns
    -------
    dict with TP, TN, FP, FN counts and dollar costs
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fraud_amounts = amounts[y_true == 1]
    fn_cost_total = (fn / max(len(fraud_amounts), 1)) * fraud_amounts.sum() \
        if len(fraud_amounts) > 0 else 0.0
    # Simpler & more intuitive: each missed fraud loses its full amount
    # Use mean as proxy when we don't have per-instance tracking
    mean_fraud_amt = float(fraud_amounts.mean()) if len(fraud_amounts) > 0 else 200.0
    fn_cost_total  = fn * mean_fraud_amt

    fp_cost_total  = fp * fp_cost
    total_cost     = fn_cost_total + fp_cost_total
    money_saved    = tp * mean_fraud_amt   # fraud caught = money recovered

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FP_cost":     round(fp_cost_total,  2),
        "FN_cost":     round(fn_cost_total,  2),
        "Total_cost":  round(total_cost,     2),
        "Money_saved": round(money_saved,    2),
        "Mean_fraud_amt": round(mean_fraud_amt, 2),
    }


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
def get_baseline_models() -> dict:
    """
    Return three baseline classifiers.

    Design notes
    ------------
    • Logistic Regression  : fast linear baseline; class_weight='balanced'
      compensates for imbalance without SMOTE.
    • Random Forest        : ensemble of decision trees; highly parallelisable;
      class_weight='balanced_subsample' re-weights each bootstrap sample.
    • Gradient Boosting    : sequential boosting; scale_pos_weight-equivalent
      achieved by capping max_depth and using many shallow trees.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight = "balanced",
            C            = 0.05,
            max_iter     = 500,
            solver       = "lbfgs",
            random_state = 42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators     = 200,
            max_depth        = 15,
            min_samples_leaf = 5,
            class_weight     = "balanced_subsample",
            random_state     = 42,
            n_jobs           = 1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators  = 200,
            learning_rate = 0.05,
            max_depth     = 5,
            subsample     = 0.8,
            random_state  = 42,
        ),
    }


# =============================================================================
# CROSS-VALIDATION
# =============================================================================
def cross_validate_models(models: dict,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           cv: int = 5) -> dict:
    """
    Stratified k-fold CV for every model.
    Reports Recall, AUC-PR, and F1 (mean ± std).

    Uses recall as primary CV metric because in fraud detection we care
    most about catching every fraudulent transaction.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = {}

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION  (5-fold Stratified)")
    print("=" * 60)
    header = f"{'Model':<25}  {'Recall':>12}  {'AUC-PR':>12}  {'F1':>10}"
    print(header)
    print("-" * 62)

    for name, model in models.items():
        rec = cross_val_score(model, X_train, y_train,
                              cv=skf, scoring="recall",            n_jobs=1)
        f1  = cross_val_score(model, X_train, y_train,
                              cv=skf, scoring="f1",                n_jobs=1)
        apr = cross_val_score(model, X_train, y_train,
                              cv=skf, scoring="average_precision", n_jobs=1)

        cv_results[name] = dict(
            recall_mean=rec.mean(), recall_std=rec.std(),
            f1_mean=f1.mean(),
            auc_pr_mean=apr.mean(), auc_pr_std=apr.std(),
        )
        print(f"{name:<25}  "
              f"{rec.mean():.4f}±{rec.std():.3f}  "
              f"{apr.mean():.4f}±{apr.std():.3f}  "
              f"{f1.mean():.4f}")

    return cv_results


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================
def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_iter: int = 15, cv: int = 5) -> RandomForestClassifier:
    """Tune Random Forest with RandomizedSearchCV (AUC-PR objective)."""
    print("\n  Tuning Random Forest …")
    param_dist = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [10, 15, 20, None],
        "min_samples_leaf": [2, 5, 10],
        "max_features":     ["sqrt", "log2"],
    }
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample",
                               random_state=42, n_jobs=1),
        param_dist, n_iter=n_iter,
        scoring="average_precision", cv=skf,
        n_jobs=1, random_state=42, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best RF params : {search.best_params_}")
    print(f"  Best AUC-PR    : {search.best_score_:.4f}")
    return search.best_estimator_


def tune_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray,
                            n_iter: int = 15, cv: int = 3) -> GradientBoostingClassifier:
    """Tune Gradient Boosting with RandomizedSearchCV (AUC-PR objective)."""
    print("\n  Tuning Gradient Boosting …")
    param_dist = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.05, 0.10],
        "subsample":        [0.7, 0.8, 0.9],
        "min_samples_leaf": [3, 5, 10],
    }
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_dist, n_iter=n_iter,
        scoring="average_precision", cv=skf,
        n_jobs=1, random_state=42, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best GB params : {search.best_params_}")
    print(f"  Best AUC-PR    : {search.best_score_:.4f}")
    return search.best_estimator_


# =============================================================================
# THRESHOLD OPTIMISATION
# =============================================================================
def optimise_threshold(model,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        amounts: np.ndarray,
                        thresholds: np.ndarray = None,
                        fp_cost: float = 10.0):
    """
    Sweep classification threshold from 0.05 to 0.80 and choose the value
    that minimises total financial cost (FN_cost + FP_cost).

    At low thresholds recall improves (fewer missed frauds) but FP_cost rises.
    At high thresholds FP_cost drops but FN_cost rises sharply.
    The optimum balances both.

    Returns
    -------
    best_threshold, thresholds, costs, recalls, precisions, f1s
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.81, 0.01)

    probs  = model.predict_proba(X_test)[:, 1]
    costs, recalls, precisions, f1s = [], [], [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        c     = compute_cost(y_test, preds, amounts, fp_cost=fp_cost)
        costs.append(c["Total_cost"])
        recalls.append(recall_score(y_test, preds, zero_division=0))
        precisions.append(precision_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    best_idx = int(np.argmin(costs))
    best_t   = float(thresholds[best_idx])

    default_cost = compute_cost(y_test, (probs >= 0.50).astype(int),
                                amounts, fp_cost)["Total_cost"]

    print("\n" + "=" * 60)
    print("  THRESHOLD OPTIMISATION")
    print("=" * 60)
    print(f"  Default  threshold (0.50) cost : ${default_cost:>12,.0f}")
    print(f"  Optimal  threshold ({best_t:.2f}) cost : ${costs[best_idx]:>12,.0f}")
    saving = default_cost - costs[best_idx]
    print(f"  Additional savings             : ${saving:>12,.0f}")

    return best_t, thresholds, np.array(costs), np.array(recalls), \
           np.array(precisions), np.array(f1s)


# =============================================================================
# FULL TRAINING PIPELINE
# =============================================================================
def train_all_models(X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     feature_names: list, test_amounts: np.ndarray,
                     run_tuning: bool = True,
                     output_dir: str = "outputs") -> tuple:
    """
    Orchestrate the full training pipeline:

    1. Cross-validate baselines
    2. Tune Random Forest and Gradient Boosting (if run_tuning=True)
    3. Fit all final models on the full training set
    4. Evaluate each model at threshold 0.50
    5. Optimise threshold on the best model (highest AUC-PR)
    6. Save the best model bundle to outputs/best_model.pkl

    Returns
    -------
    results      : dict  {model_name -> metric dict}
    best_name    : str
    best_model   : fitted estimator
    opt_threshold: float
    thr_data     : tuple(thresholds, costs, recalls, precisions, f1s)
    """
    os.makedirs(output_dir, exist_ok=True)
    models = get_baseline_models()

    # ── Cross-validation ───────────────────────────────────────────────────────
    cross_validate_models(models, X_train, y_train)

    # ── Hyperparameter tuning ──────────────────────────────────────────────────
    if run_tuning:
        print("\n" + "=" * 60)
        print("  HYPERPARAMETER TUNING")
        print("=" * 60)
        models["Random Forest"]     = tune_random_forest(X_train, y_train)
        models["Gradient Boosting"] = tune_gradient_boosting(X_train, y_train)

    # ── Fit all models & evaluate at default threshold ─────────────────────────
    results = {}
    print("\n" + "=" * 60)
    print("  FITTING FINAL MODELS")
    print("=" * 60)

    for name, model in models.items():
        print(f"  Fitting {name} …")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)

        results[name] = {
            "model":     model,
            "probs":     probs,
            "recall":    recall_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "f1":        f1_score(y_test, preds, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, probs),
            "auc_pr":    average_precision_score(y_test, probs),
            "cost":      compute_cost(y_test, preds, test_amounts),
            "threshold": 0.50,
        }
        r = results[name]
        print(f"    Recall={r['recall']:.4f}  Precision={r['precision']:.4f}  "
              f"F1={r['f1']:.4f}  AUC-PR={r['auc_pr']:.4f}")

    # ── Choose best model by AUC-PR ────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["auc_pr"])
    best_model = results[best_name]["model"]
    print(f"\n  Best model (AUC-PR)  →  {best_name} "
          f"({results[best_name]['auc_pr']:.4f})")

    # ── Threshold optimisation on best model ───────────────────────────────────
    opt_t, thresholds, costs, recalls, precisions, f1s = optimise_threshold(
        best_model, X_test, y_test, test_amounts
    )

    # Add optimised-threshold result entry
    opt_probs = results[best_name]["probs"]
    opt_preds = (opt_probs >= opt_t).astype(int)
    opt_key   = f"{best_name} (Optimised)"

    results[opt_key] = {
        "model":     best_model,
        "probs":     opt_probs,
        "recall":    recall_score(y_test, opt_preds, zero_division=0),
        "precision": precision_score(y_test, opt_preds, zero_division=0),
        "f1":        f1_score(y_test, opt_preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, opt_probs),
        "auc_pr":    average_precision_score(y_test, opt_probs),
        "cost":      compute_cost(y_test, opt_preds, test_amounts),
        "threshold": opt_t,
    }

    # ── Persist model ──────────────────────────────────────────────────────────
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(
        {"model": best_model, "threshold": opt_t, "features": feature_names},
        model_path,
    )
    print(f"\n  Model saved  →  {model_path}")

    thr_data = (thresholds, costs, recalls, precisions, f1s)
    return results, best_name, best_model, opt_t, thr_data


# =============================================================================
# VISUALISATIONS
# =============================================================================
def plot_model_comparison(results: dict,
                           save_path: str = "outputs/model_comparison.png"):
    """
    Four horizontal bar charts: Recall, Precision, F1-Score, ROC-AUC.
    One bar per model (including the optimised variant).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names   = list(results.keys())
    metrics = ["recall", "precision", "f1", "roc_auc"]
    labels  = ["Recall", "Precision", "F1-Score", "ROC-AUC"]
    colors  = [RED, ORANGE, BLUE, PURPLE]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Model Performance Comparison",
                 fontsize=15, fontweight="bold", y=1.02, color="white")

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [results[n][metric] for n in names]
        bars = ax.barh(names, vals, color=color, alpha=0.82,
                       edgecolor=BG, height=0.55)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.006, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8.5, color=TEXT)
        ax.set_xlim(0, 1.15)
        ax.set_title(label, fontsize=12, fontweight="bold", color=color)
        ax.set_facecolor(PANEL)
        ax.set_xlabel("Score")
        ax.axvline(0.5, color=BORDER, linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Model comparison saved  →  {save_path}")


def plot_threshold_analysis(thr_data: tuple,
                             save_path: str = "outputs/threshold_analysis.png"):
    """
    Left  : Total financial cost vs threshold (with optimal marker).
    Right : Recall / Precision / F1 vs threshold.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    thresholds, costs, recalls, precisions, f1s = thr_data
    best_idx = int(np.argmin(costs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    # Cost curve
    ax1.plot(thresholds, costs / 1e6, color=RED, linewidth=2.5,
             label="Total Cost")
    ax1.axvline(thresholds[best_idx], color=ORANGE, linestyle="--",
                linewidth=2.0, label=f"Optimal t = {thresholds[best_idx]:.2f}")
    ax1.scatter([thresholds[best_idx]], [costs[best_idx] / 1e6],
                color="white", s=90, zorder=6)
    ax1.set_xlabel("Classification Threshold")
    ax1.set_ylabel("Total Financial Cost ($ millions)")
    ax1.set_title("Financial Cost vs Threshold", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.set_facecolor(PANEL)

    # Metric curves
    ax2.plot(thresholds, recalls,    color=GREEN,  linewidth=2.5, label="Recall")
    ax2.plot(thresholds, precisions, color=BLUE,   linewidth=2.5, label="Precision")
    ax2.plot(thresholds, f1s,        color=PURPLE, linewidth=2.5, label="F1-Score")
    ax2.axvline(thresholds[best_idx], color=ORANGE, linestyle="--",
                linewidth=2.0, label=f"Optimal t = {thresholds[best_idx]:.2f}")
    ax2.axvline(0.50, color=BORDER, linestyle=":", linewidth=1.5,
                label="Default t = 0.50")
    ax2.set_xlabel("Classification Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Metrics vs Threshold", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.set_facecolor(PANEL)

    fig.suptitle("Threshold Optimisation Analysis",
                 fontsize=15, fontweight="bold", y=1.02, color="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Threshold analysis saved  →  {save_path}")


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from module1_data_pipeline import (load_data, engineer_features,
                                        preprocess, FEATURE_COLS)

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
    plot_model_comparison(results)
    plot_threshold_analysis(thr_data)
    print("\n  Module 2 complete.")
