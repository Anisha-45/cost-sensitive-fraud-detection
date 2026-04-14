# =============================================================================
# MODULE 1 — Data Pipeline & Preprocessing
# =============================================================================
# Responsibilities:
#   • Load fraudTrain.csv / fraudTest.csv
#   • Clean & validate raw data
#   • Feature engineering (temporal, geo-distance, age, category encoding)
#   • Handle class imbalance with SMOTE (manual, no imblearn required)
#   • Scale features with RobustScaler
#   • Produce exploratory data-analysis figure
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ── Plot theme ─────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d2e"
BORDER  = "#3a3d52"
TEXT    = "#c9d1d9"
MUTED   = "#8b949e"
RED     = "#ff4757"
GREEN   = "#2ed573"
ORANGE  = "#ffa502"
BLUE    = "#1e90ff"
PURPLE  = "#a29bfe"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":  PANEL,
    "axes.edgecolor":   BORDER,"axes.labelcolor": TEXT,
    "xtick.color":      MUTED, "ytick.color":     MUTED,
    "text.color":       TEXT,  "grid.color":      BORDER,
    "grid.linestyle":   "--",  "grid.alpha":      0.4,
    "legend.facecolor": PANEL, "legend.edgecolor": BORDER,
})


# =============================================================================
# MANUAL SMOTE  (k-NN interpolation, no external dependency)
# =============================================================================
class ManualSMOTE:
    """
    Synthetic Minority Oversampling Technique.

    For each existing fraud sample, finds k nearest fraud neighbours and
    generates synthetic points along the line segments between them.

    Parameters
    ----------
    sampling_strategy : float
        Desired ratio of minority / majority samples after resampling.
    k_neighbors : int
        Number of nearest neighbours to use.
    random_state : int
    """
    def __init__(self, sampling_strategy: float = 0.20,
                 k_neighbors: int = 5, random_state: int = 42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors       = k_neighbors
        self.rng               = np.random.default_rng(random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        X_min = X[y == 1]
        n_maj = int((y == 0).sum())
        n_target_min = int(n_maj * self.sampling_strategy)
        n_syn = max(0, n_target_min - len(X_min))

        if n_syn == 0:
            return X, y

        k = min(self.k_neighbors, len(X_min) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1)
        nn.fit(X_min)
        _, indices = nn.kneighbors(X_min)   # shape (n_min, k+1), col-0 = self

        synthetic = []
        for _ in range(n_syn):
            i       = self.rng.integers(0, len(X_min))
            j       = self.rng.choice(indices[i][1:])
            alpha   = self.rng.random()
            syn_row = X_min[i] + alpha * (X_min[j] - X_min[i])
            synthetic.append(syn_row)

        X_syn = np.array(synthetic)
        y_syn = np.ones(len(X_syn), dtype=int)

        X_out = np.vstack([X, X_syn])
        y_out = np.concatenate([y, y_syn])

        perm = self.rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]


# =============================================================================
# 1.  LOAD DATA
# =============================================================================
def load_data(train_path: str, test_path: str):
    """
    Load the Kaggle credit-card fraud CSVs.

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    print("=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    df_train = pd.read_csv(train_path, index_col=0).sample(100000)
    df_test  = pd.read_csv(test_path, index_col=0).sample(50000)

    print(f"  Train : {df_train.shape[0]:>10,} rows | "
          f"Fraud: {df_train['is_fraud'].sum():,} "
          f"({100*df_train['is_fraud'].mean():.2f}%)")
    print(f"  Test  : {df_test.shape[0]:>10,} rows | "
          f"Fraud: {df_test['is_fraud'].sum():,} "
          f"({100*df_test['is_fraud'].mean():.2f}%)")
    return df_train, df_test


# =============================================================================
# 2.  FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model features from raw transaction columns.

    New features created
    --------------------
    trans_hour      : hour of day (0-23) — fraud peaks late-night
    trans_dow       : day of week (0=Mon … 6=Sun)
    trans_month     : month (1-12)
    is_weekend      : 1 if Sat/Sun
    is_night        : 1 if 22:00–05:59
    age             : cardholder age at transaction time
    distance_km     : Haversine distance (cardholder ↔ merchant)
    amt_log         : log1p(transaction amount) — reduces skew
    amt_zscore      : per-category z-score of amount
    category_enc    : label-encoded merchant category
    gender_enc      : 0=F, 1=M
    city_pop_log    : log1p(city_pop)
    """
    df = df.copy()

    # ── datetime parsing ───────────────────────────────────────────────────────
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"]                   = pd.to_datetime(df["dob"])

    # ── temporal features ──────────────────────────────────────────────────────
    df["trans_hour"]  = df["trans_date_trans_time"].dt.hour
    df["trans_dow"]   = df["trans_date_trans_time"].dt.dayofweek
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"]  = (df["trans_dow"] >= 5).astype(int)
    df["is_night"]    = ((df["trans_hour"] >= 22) | (df["trans_hour"] <= 5)).astype(int)

    # ── age at transaction ─────────────────────────────────────────────────────
    df["age"] = ((df["trans_date_trans_time"] - df["dob"])
                 .dt.days / 365.25).astype(int)

    # ── Haversine distance (cardholder home ↔ merchant) ───────────────────────
    R = 6371.0    # Earth radius km
    lat1 = np.radians(df["lat"].values)
    lon1 = np.radians(df["long"].values)
    lat2 = np.radians(df["merch_lat"].values)
    lon2 = np.radians(df["merch_long"].values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    df["distance_km"] = 2 * R * np.arcsin(np.sqrt(a))

    # ── amount transformations ─────────────────────────────────────────────────
    df["amt_log"] = np.log1p(df["amt"])

    # Per-category z-score (flag unusually large spend within a category)
    cat_stats = df.groupby("category")["amt"].transform
    df["amt_zscore"] = ((df["amt"] - cat_stats("mean")) /
                        (cat_stats("std").replace(0, 1)))

    # ── categorical encodings ──────────────────────────────────────────────────
    le_cat = LabelEncoder()
    df["category_enc"] = le_cat.fit_transform(df["category"])

    df["gender_enc"] = (df["gender"] == "M").astype(int)

    # ── city population ────────────────────────────────────────────────────────
    df["city_pop_log"] = np.log1p(df["city_pop"])

    return df


# =============================================================================
# 3.  SELECT MODEL FEATURES
# =============================================================================
FEATURE_COLS = [
    "amt",
    "amt_log",
    "amt_zscore",
    "trans_hour",
    "trans_dow",
    "trans_month",
    "is_weekend",
    "is_night",
    "age",
    "distance_km",
    "city_pop_log",
    "category_enc",
    "gender_enc",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]

TARGET_COL = "is_fraud"


def get_X_y(df: pd.DataFrame):
    """Return feature matrix and target vector."""
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(int)
    return X, y


# =============================================================================
# 4.  PREPROCESS  (scale + optional SMOTE on training set only)
# =============================================================================
def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame,
               apply_smote: bool = True, smote_strategy: float = 0.20,
               random_state: int = 42):
    """
    Scale features with RobustScaler (fit on train, apply to test).
    Optionally apply SMOTE to the training set to address class imbalance.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING")
    print("=" * 60)

    X_train_raw, y_train = get_X_y(df_train)
    X_test_raw,  y_test  = get_X_y(df_test)

    # ── RobustScaler: robust to outliers in 'amt' ──────────────────────────────
    scaler  = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    print(f"  Train before SMOTE : {X_train.shape[0]:>10,} rows | "
          f"Fraud: {y_train.sum():,}")
    print(f"  Test               : {X_test.shape[0]:>10,} rows | "
          f"Fraud: {y_test.sum():,}")

    # ── SMOTE only on training split ───────────────────────────────────────────
    if apply_smote:
        print(f"\n  Applying SMOTE (strategy={smote_strategy}) …")
        smote = ManualSMOTE(
            sampling_strategy=smote_strategy,
            k_neighbors=5,
            random_state=random_state,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  Train after SMOTE  : {X_train.shape[0]:>10,} rows | "
              f"Fraud: {y_train.sum():,}")

    return X_train, X_test, y_train, y_test, scaler


# =============================================================================
# 5.  EXPLORATORY DATA ANALYSIS
# =============================================================================
def run_eda(df_train: pd.DataFrame,
            save_path: str = "outputs/eda_overview.png"):
    """
    Generate a 6-panel EDA figure and save to disk.

    Panels
    ------
    1. Class distribution (donut)
    2. Transaction amount by class
    3. Fraud by hour of day
    4. Fraud rate by merchant category
    5. Age distribution by class
    6. Geographic distance by class
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    counts = df_train["is_fraud"].value_counts()

    # ── 1. Class donut ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie(
        counts,
        labels   = ["Legitimate", "Fraud"],
        colors   = [GREEN, RED],
        autopct  = "%1.2f%%",
        startangle = 90,
        wedgeprops = {"width": 0.46, "edgecolor": BG, "linewidth": 2},
        textprops  = {"color": TEXT, "fontsize": 11},
    )
    ax1.set_title("Class Distribution", fontsize=13, fontweight="bold", pad=14)

    # ── 2. Amount distribution ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for cls, color, lbl in [(0, GREEN, "Legitimate"), (1, RED, "Fraud")]:
        vals = df_train[df_train["is_fraud"] == cls]["amt"].clip(upper=500)
        ax2.hist(vals, bins=60, alpha=0.65, color=color,
                 label=lbl, density=True, edgecolor="none")
    ax2.set_xlabel("Transaction Amount ($)")
    ax2.set_ylabel("Density")
    ax2.set_title("Transaction Amount Distribution", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)

    # ── 3. Fraud by hour ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    hourly = (df_train.groupby(["trans_hour", "is_fraud"])
              .size().unstack(fill_value=0))
    hourly["fraud_rate"] = hourly[1] / (hourly[0] + hourly[1]) * 100
    ax3.bar(hourly.index, hourly["fraud_rate"], color=ORANGE, alpha=0.85,
            edgecolor=BG, width=0.8)
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Fraud Rate (%)")
    ax3.set_title("Fraud Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax3.set_xticks(range(0, 24, 2))

    # ── 4. Fraud rate by category ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    cat_fraud = (df_train.groupby("category")["is_fraud"]
                 .mean() * 100).sort_values(ascending=True)
    bars = ax4.barh(cat_fraud.index, cat_fraud.values, alpha=0.85,
                    color=[RED if v > cat_fraud.median() else BLUE
                           for v in cat_fraud.values],
                    edgecolor=BG)
    ax4.set_xlabel("Fraud Rate (%)")
    ax4.set_title("Fraud Rate by Category", fontsize=13, fontweight="bold")
    ax4.tick_params(axis="y", labelsize=9)
    for bar, val in zip(bars, cat_fraud.values):
        ax4.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", fontsize=8, color=TEXT)

    # ── 5. Age distribution ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for cls, color, lbl in [(0, GREEN, "Legitimate"), (1, RED, "Fraud")]:
        vals = df_train[df_train["is_fraud"] == cls]["age"].clip(18, 90)
        ax5.hist(vals, bins=40, alpha=0.65, color=color,
                 label=lbl, density=True, edgecolor="none")
    ax5.set_xlabel("Cardholder Age")
    ax5.set_ylabel("Density")
    ax5.set_title("Age Distribution", fontsize=13, fontweight="bold")
    ax5.legend(fontsize=10)

    # ── 6. Distance distribution ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    for cls, color, lbl in [(0, GREEN, "Legitimate"), (1, RED, "Fraud")]:
        vals = df_train[df_train["is_fraud"] == cls]["distance_km"].clip(upper=1000)
        ax6.hist(vals, bins=50, alpha=0.65, color=color,
                 label=lbl, density=True, edgecolor="none")
    ax6.set_xlabel("Cardholder ↔ Merchant Distance (km)")
    ax6.set_ylabel("Density")
    ax6.set_title("Transaction Distance Distribution", fontsize=13, fontweight="bold")
    ax6.legend(fontsize=10)

    fig.suptitle(
        "Credit Card Fraud Detection — Exploratory Data Analysis",
        fontsize=17, fontweight="bold", y=1.01, color="white",
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  EDA figure saved  →  {save_path}")


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    TRAIN_PATH = "/mnt/user-data/uploads/fraudTrain.csv"
    TEST_PATH  = "/mnt/user-data/uploads/fraudTest.csv"

    df_train_raw, df_test_raw = load_data(TRAIN_PATH, TEST_PATH)

    print("\n  Engineering features …")
    df_train = engineer_features(df_train_raw)
    df_test  = engineer_features(df_test_raw)

    run_eda(df_train, save_path="outputs/eda_overview.png")

    X_train, X_test, y_train, y_test, scaler = preprocess(
        df_train, df_test, apply_smote=True, smote_strategy=0.20
    )

    print(f"\n  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print("\n  Module 1 complete.")
