"""
Lab 11: Outlier Detection and Ensemble Methods
Topics:
  Z-score method
  IQR method
  Isolation Forest
  Random Forest classifier
  Feature importance analysis
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
DATA_FILE = DATA_DIR / "credit_card_transactions.csv"
CLASS_NAMES = ["Legitimate", "Fraud"]


def section(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  figure saved -> {path.name}")


print("=" * 78)
print("LAB 11 : OUTLIER DETECTION AND ENSEMBLE METHODS")
print("=" * 78)


# ==================================================================================
# 1. LOADING THE DATA
# ==================================================================================
section("1. LOADING THE DATA")

df = pd.read_csv(DATA_FILE).drop_duplicates().dropna().reset_index(drop=True)
categorical = ["Merchant_Category", "Card_Type", "Country"]
df_encoded = pd.concat([df.drop(columns=categorical),
                         pd.get_dummies(df[categorical], dtype=int)], axis=1)

numeric_cols = [c for c in df.columns if c not in categorical and c != "Class"]
feature_cols = [c for c in df_encoded.columns if c != "Class"]

X = df_encoded[feature_cols]
y = df_encoded["Class"]
print(f"Dataset shape : {df.shape}")
print(f"Class distribution :\n{y.value_counts().to_string()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================================================================================
# 2. Z-SCORE METHOD
# ==================================================================================
section("2. Z-SCORE METHOD")

print("  z = (x - mean(x)) / standard_deviation(x)")
print("  |z| > 3 is treated as an outlier (99.7 % of a normal distribution)")
print("  modified z-score = 0.6745 * (x - median) / MAD  (robust version)")


def z_score(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def modified_z_score(series: pd.Series) -> pd.Series:
    """Robust z-score based on the median and the median absolute deviation."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        mad = series.std() or 1.0
    return 0.6745 * (series - median) / mad


z_scores = pd.DataFrame({c: z_score(df[c]) for c in numeric_cols})
z_outliers = (z_scores.abs() > 3)

print("Number of outliers (|z| > 3) per feature:")
outlier_counts = z_outliers.sum().sort_values(ascending=False)
print(outlier_counts[outlier_counts > 0].to_string())

print(f"\nTotal outliers found with the z-score method : {int(z_outliers.sum().sum())}")
print(f"Transactions containing at least one outlier   : {int(z_outliers.any(axis=1).sum())}")

amount = df["Amount"]
mz = modified_z_score(amount)
mz_outliers = mz.abs() > 3.5
print(f"\nAmount outliers with the standard z-score   : {int((z_score(amount).abs() > 3).sum())}")
print(f"Amount outliers with the modified z-score   : {int(mz_outliers.sum())}")
print(f"Transactions flagged by both methods        : "
      f"{int(((z_score(amount).abs() > 3) & mz_outliers).sum())}")

print("\nThe 10 largest transactions according to the modified z-score:")
top_outliers = df.loc[mz.nlargest(10).index, ["Time", "Amount", "Merchant_Category", "Class"]]
top_outliers = top_outliers.assign(modified_z=mz.nlargest(10).round(3).values)
print(top_outliers.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(z_score(amount), bins=70, color="#4C72B0", edgecolor="black")
ax.axvspan(-3, 3, color="green", alpha=0.15, label="normal range (-3, 3)")
ax.axvline(-3, color="crimson", linestyle="--")
ax.axvline(3, color="crimson", linestyle="--")
ax.set_title("Distribution of the z-scores of the Amount column")
ax.set_xlabel("z-score")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
save(fig, "lab11_zscore.png")


# ==================================================================================
# 3. IQR METHOD
# ==================================================================================
section("3. IQR METHOD")

print("  IQR = Q3 - Q1")
print("  lower bound = Q1 - 1.5 * IQR      upper bound = Q3 + 1.5 * IQR")
print("  every value outside the bounds is an outlier (factor 3.0 = extreme outliers)")


def iqr_bounds(series: pd.Series, factor: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr, iqr


iqr_mask = pd.Series(False, index=df.index)
bounds = {}
for col in numeric_cols:
    low, high, iqr = iqr_bounds(df[col])
    bounds[col] = (low, high)
    iqr_mask |= (df[col] < low) | (df[col] > high)

print("IQR bounds of the most interesting features:")
for col in ["Amount", "Time", "V14", "V17"]:
    low, high = bounds[col]
    print(f"  {col:10} lower = {low:12.3f}   upper = {high:12.3f}")

print(f"\nTransactions with at least one IQR outlier  : {int(iqr_mask.sum())} "
      f"({iqr_mask.mean() * 100:.2f} %)")
print(f"Transactions flagged by the z-score method  : {int(z_outliers.any(axis=1).sum())}")
print(f"Flagged by both methods                    : {int((iqr_mask & z_outliers.any(axis=1)).sum())}")

low3, high3, _ = iqr_bounds(amount, factor=3.0)
extreme = (amount < low3) | (amount > high3)
print(f"\nExtreme Amount outliers (factor 3.0) : {int(extreme.sum())} "
      f"(bounds {low3:.2f} and {high3:.2f})")

print("\nExample of IQR outliers:")
print(df.loc[iqr_mask, ["Time", "Amount", "Merchant_Category", "Class"]].head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.boxplot(x=df["Amount"], ax=ax, flierprops=dict(marker="^", markerfacecolor="crimson",
                                                    markersize=6, alpha=0.7))
ax.set_title("Transaction Amount - box plot with the IQR outliers")
plt.tight_layout()
save(fig, "lab11_iqr_boxplot.png")

comparison = pd.DataFrame({
    "Z-score outliers": z_outliers.any(axis=1),
    "IQR outliers": iqr_mask,
})
print("\nHow the two methods agree:")
print(pd.crosstab(comparison["Z-score outliers"], comparison["IQR outliers"]).to_string())


# ==================================================================================
# 4. ISOLATION FOREST
# ==================================================================================
section("4. ISOLATION FOREST")

print("A forest of random trees; an anomaly is isolated after very few splits, so")
print("it gets a short average path length in the forest.")
print("  contamination : expected ratio of outliers (0.01 = 1 %)")
print("  n_estimators  : number of trees")
print("  Output        : -1 = outlier (anomaly), 1 = inlier")


iso_forest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso_predictions = iso_forest.fit_predict(X_scaled)
outlier_mask = pd.Series(iso_predictions == -1, index=df.index)

print(f"Outliers detected by the Isolation Forest : {int(outlier_mask.sum())} "
      f"({outlier_mask.mean() * 100:.2f} %)")
print(f"Transactions flagged by the IQR method    : {int(iqr_mask.sum())}")
print(f"Flagged by both methods                   : {int((outlier_mask & iqr_mask).sum())}")
print(f"Detected only by the Isolation Forest     : {int((outlier_mask & ~iqr_mask).sum())}")
print(f"Detected only by the IQR method           : {int((~outlier_mask & iqr_mask).sum())}")

print("\nRelation between the detected outliers and the fraud label:")
print(pd.crosstab(outlier_mask.map({True: "Outlier", False: "Inlier"}),
                  df["Class"].map({0: "Legitimate", 1: "Fraud"})).to_string())

anomaly_scores = iso_forest.score_samples(X_scaled)
top_idx = np.argsort(anomaly_scores)[:10]
print("\nThe 10 most anomalous transactions:")
print(df.loc[top_idx, ["Time", "Amount", "Merchant_Category", "Class"]]
      .assign(anomaly_score=anomaly_scores[top_idx].round(4)).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(df["Time"], df["Amount"], s=6, color="#4C72B0", alpha=0.4, label="inlier")
axes[0].scatter(df.loc[outlier_mask, "Time"], df.loc[outlier_mask, "Amount"],
                s=35, color="crimson", marker="^", label="Isolation Forest outlier")
axes[0].set_title("Outliers detected by the Isolation Forest")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Amount")
axes[0].legend()

axes[1].scatter(df["V14"], df["V17"], s=6, color="#4C72B0", alpha=0.4, label="inlier")
axes[1].scatter(df.loc[outlier_mask, "V14"], df.loc[outlier_mask, "V17"],
                s=35, color="crimson", marker="^", label="Isolation Forest outlier")
axes[1].set_title("Outliers in the V14 / V17 space")
axes[1].set_xlabel("V14")
axes[1].set_ylabel("V17")
axes[1].legend()
plt.tight_layout()
save(fig, "lab11_isolation_forest.png")


# ==================================================================================
# 5. RANDOM FOREST CLASSIFIER
# ==================================================================================
section("5. RANDOM FOREST CLASSIFIER")

print("Many decision trees are built on bootstrap samples of the data and on random")
print("subsets of the features; the final prediction is the majority vote.")
print("  class_weight = balanced_subsample compensates the strong class imbalance")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1,
                            class_weight="balanced_subsample", random_state=42,
                            n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("Random Forest performance on the test set:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1 score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  ROC AUC   : {roc_auc_score(y_test, y_prob):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

dt = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nComparison: Decision Tree vs Random Forest")
comparison_table = pd.DataFrame({
    "Decision Tree": {
        "Accuracy": accuracy_score(y_test, dt_pred),
        "Precision": precision_score(y_test, dt_pred, zero_division=0),
        "Recall": recall_score(y_test, dt_pred, zero_division=0),
        "F1": f1_score(y_test, dt_pred, zero_division=0),
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    },
})
print(comparison_table.round(4).to_string())

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Random Forest")
plt.tight_layout()
save(fig, "lab11_confusion_matrix.png")


# ==================================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ==================================================================================
section("6. FEATURE IMPORTANCE ANALYSIS")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Top 15 features (impurity based importance of the Random Forest):")
print(importances.head(15).round(5).to_string())

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(importances.head(15).index[::-1], importances.head(15).values[::-1],
        color="#4C72B0", edgecolor="black")
ax.set_xlabel("Feature importance")
ax.set_title("Top 15 features - Random Forest")
plt.tight_layout()
save(fig, "lab11_feature_importance.png")

print("\nPermutation importance (measured on the test set):")
perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42,
                              scoring="f1", n_jobs=-1)
perm_df = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print(perm_df.head(15).round(5).to_string())

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(perm_df.head(15).index[::-1], perm_df.head(15).values[::-1],
        color="#C44E52", edgecolor="black")
ax.set_xlabel("Mean decrease in F1 score when the feature is shuffled")
ax.set_title("Top 15 features - permutation importance")
plt.tight_layout()
save(fig, "lab11_permutation_importance.png")

top_feature = importances.index[0]
print(f"\nThe most important feature is '{top_feature}'.")
print(f"Average {top_feature}: legitimate = {df.loc[df['Class'] == 0, top_feature].mean():.4f}, "
      f"fraud = {df.loc[df['Class'] == 1, top_feature].mean():.4f}")

importance_df = pd.DataFrame({
    "impurity_importance": importances,
    "permutation_importance": perm_df,
}).sort_values("impurity_importance", ascending=False)
importance_df.to_csv(OUT_DIR / "lab11_feature_importance.csv")
print(f"\nFeature importance table saved -> {OUT_DIR / 'lab11_feature_importance.csv'}")

print("\n" + "=" * 78)
print("END OF LAB 11")
print("=" * 78)
