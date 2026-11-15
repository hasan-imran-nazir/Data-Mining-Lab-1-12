"""
Lab 3: Data Visualization and Exploratory Data Analysis (EDA)
Topics:
  Histograms
  Bar charts
  Scatter plots
  Box plots
  Correlation matrix and heatmap
  Identifying outliers and relationships
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

DATA_FILE = DATA_DIR / "credit_card_transactions.csv"

LABEL = {0: "Legitimate", 1: "Fraud"}
COLOR = {0: "#4C72B0", 1: "#C44E52"}
PALETTE = {"Legitimate": "#4C72B0", "Fraud": "#C44E52"}


def section(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def save(fig, name: str) -> None:
    """Save a figure and close it."""
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  figure saved -> {path.name}")


print("=" * 78)
print("LAB 3 : DATA VISUALIZATION AND EXPLORATORY DATA ANALYSIS")
print("=" * 78)

# ----------------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna(subset=["Amount"]).reset_index(drop=True)
df["Class_Name"] = df["Class"].map(LABEL)

print(f"Dataset : {DATA_FILE}")
print(f"Shape   : {df.shape}")
print(df["Class_Name"].value_counts())
print(f"Fraud rate : {df['Class'].mean() * 100:.3f}%")


# ==================================================================================
# 1. HISTOGRAMS
# ==================================================================================
section("1. HISTOGRAMS")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(df["Amount"], bins=60, color="#4C72B0", edgecolor="black", alpha=0.85)
axes[0].set_title("Distribution of Transaction Amount")
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Frequency")

axes[1].hist(df["Time"], bins=60, color="#55A868", edgecolor="black", alpha=0.85)
axes[1].set_title("Distribution of Time (seconds)")
axes[1].set_xlabel("Time")

axes[2].hist(df["V14"], bins=60, color="#C44E52", edgecolor="black", alpha=0.85)
axes[2].set_title("Distribution of V14 (informative feature)")
axes[2].set_xlabel("V14")

plt.tight_layout()
save(fig, "lab3_1_histograms.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for cls, name in LABEL.items():
    subset = df[df["Class"] == cls]
    axes[0].hist(subset["Amount"], bins=50, alpha=0.6, label=name, color=COLOR[cls])
    axes[1].hist(subset["V14"], bins=50, alpha=0.6, label=name, color=COLOR[cls])
axes[0].set_title("Amount: Legitimate vs Fraud")
axes[1].set_title("V14: Legitimate vs Fraud")
for ax in axes:
    ax.legend()
plt.tight_layout()
save(fig, "lab3_1b_histogram_by_class.png")

print("\nSkewness and kurtosis of the main numeric columns:")
skew = df[["Time", "Amount", "V12", "V14", "V17"]].skew().round(3)
kurt = df[["Time", "Amount", "V12", "V14", "V17"]].kurtosis().round(3)
print(pd.DataFrame({"Skewness": skew, "Kurtosis": kurt}))


# ==================================================================================
# 2. BAR CHARTS
# ==================================================================================
section("2. BAR CHARTS")

counts = df["Merchant_Category"].value_counts()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(counts.index, counts.values, color=sns.color_palette("deep"), edgecolor="black")
for i, v in enumerate(counts.values):
    ax.text(i, v + 120, f"{v}", ha="center", fontsize=9)
ax.set_title("Number of Transactions per Merchant Category")
ax.set_xlabel("Merchant Category")
ax.set_ylabel("Number of Transactions")
plt.tight_layout()
save(fig, "lab3_2_bar_merchant.png")
print("\nTransactions per merchant category:")
print(counts)

fig, ax = plt.subplots(figsize=(6, 4))
class_counts = df["Class_Name"].value_counts()
ax.bar(class_counts.index, class_counts.values, color=[COLOR[0], COLOR[1]], edgecolor="black")
ax.set_yscale("log")
ax.set_title("Class Distribution (log scale)")
ax.set_ylabel("Number of Transactions")
for i, v in enumerate(class_counts.values):
    ax.text(i, v * 1.2, f"{v}", ha="center")
plt.tight_layout()
save(fig, "lab3_2b_bar_class.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.countplot(data=df, x="Country", hue="Class_Name", palette=PALETTE, ax=axes[0])
axes[0].set_title("Transactions per Country by Class")
axes[0].tick_params(axis="x", rotation=45)
sns.countplot(data=df, x="Card_Type", hue="Class_Name", palette=PALETTE, ax=axes[1])
axes[1].set_title("Transactions per Card Type by Class")
plt.tight_layout()
save(fig, "lab3_2c_bar_country_card.png")

fig, ax = plt.subplots(figsize=(9, 4.5))
pd.crosstab(df["Merchant_Category"], df["Class_Name"]).plot(
    kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#C44E52"], edgecolor="black")
ax.set_title("Merchant Category vs Class (stacked)")
plt.tight_layout()
save(fig, "lab3_2d_bar_stacked.png")


# ==================================================================================
# 3. SCATTER PLOTS
# ==================================================================================
section("3. SCATTER PLOTS")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cls, name in LABEL.items():
    subset = df[df["Class"] == cls]
    axes[0].scatter(subset["Time"], subset["Amount"], s=4, alpha=0.4,
                    label=name, color=COLOR[cls])
axes[0].set_title("Transaction Amount vs Time")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amount")
axes[0].legend()

for cls, name in LABEL.items():
    subset = df[df["Class"] == cls]
    axes[1].scatter(subset["V14"], subset["V17"], s=6, alpha=0.5,
                    label=name, color=COLOR[cls])
axes[1].set_title("V14 vs V17 (strongest fraud indicators)")
axes[1].set_xlabel("V14")
axes[1].set_ylabel("V17")
axes[1].legend()
plt.tight_layout()
save(fig, "lab3_3_scatter.png")

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(data=df, x="V14", y="V17", hue="Class_Name", palette=PALETTE,
                s=6, alpha=0.5, ax=ax)
sns.regplot(data=df[df["Class"] == 0], x="V14", y="V17", scatter=False,
            color="steelblue", ax=ax)
ax.set_title("Relationship between V14 and V17")
plt.tight_layout()
save(fig, "lab3_3b_scatter_regression.png")

print("\nPearson correlation of the top pairs:")
print(df[["Time", "Amount", "V12", "V14", "V17", "Class"]].corr().round(3))


# ==================================================================================
# 4. BOX PLOTS
# ==================================================================================
section("4. BOX PLOTS")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
sns.boxplot(data=df, x="Class_Name", y="Amount", palette=PALETTE, ax=axes[0])
axes[0].set_title("Amount by Class")
sns.boxplot(data=df, x="Class_Name", y="V14", palette=PALETTE, ax=axes[1])
axes[1].set_title("V14 by Class")
sns.boxplot(data=df, x="Merchant_Category", y="Amount", ax=axes[2])
axes[2].set_title("Amount by Merchant Category")
axes[2].tick_params(axis="x", rotation=45)
plt.tight_layout()
save(fig, "lab3_4_boxplots.png")

fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=df, x="Country", y="Amount", ax=ax)
ax.set_title("Amount distribution per Country")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save(fig, "lab3_4b_box_country.png")


# ==================================================================================
# 5. CORRELATION MATRIX AND HEATMAP
# ==================================================================================
section("5. CORRELATION MATRIX AND HEATMAP")

corr_cols = ["Time", "Amount", "V7", "V12", "V14", "V17", "Class"]
corr_matrix = df[corr_cols].corr()

print("Correlation matrix:")
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap (selected features)")
plt.tight_layout()
save(fig, "lab3_5_heatmap.png")

full_corr = df.select_dtypes(include=["number"]).corr()
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(full_corr, cmap="viridis", xticklabels=12, yticklabels=12,
            square=True, ax=ax)
ax.set_title("Correlation Heatmap - all numeric features")
plt.tight_layout()
save(fig, "lab3_5b_heatmap_full.png")

target_corr = df.select_dtypes(include=["number"]).corr()["Class"].drop("Class").sort_values()
print("\n10 strongest NEGATIVE correlations with Class:")
print(target_corr.head(10).round(4))
print("\n10 strongest POSITIVE correlations with Class:")
print(target_corr.tail(10).round(4))


# ==================================================================================
# 6. IDENTIFYING OUTLIERS AND RELATIONSHIPS
# ==================================================================================
section("6. IDENTIFYING OUTLIERS AND RELATIONSHIPS")

print("IQR rule: IQR = Q3 - Q1, a value is an outlier when it is outside")
print("[Q1 - 1.5*IQR, Q3 + 1.5*IQR].  Z-score rule: |z| > 3.")

def iqr_outliers(series: pd.Series, factor: float = 1.5):
    """Return (lower_bound, upper_bound, outlier_mask) using the IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    mask = (series < low) | (series > high)
    return low, high, mask


amount_low, amount_high, amount_outliers = iqr_outliers(df["Amount"])
print(f"Amount  Q1 - 1.5*IQR lower bound : {amount_low:.2f}")
print(f"Amount  Q3 + 1.5*IQR upper bound : {amount_high:.2f}")
print(f"Number of amount outliers        : {int(amount_outliers.sum())} "
      f"({amount_outliers.mean() * 100:.2f}%)")

print("\nExample outlier transactions:")
print(df.loc[amount_outliers, ["Time", "Amount", "Merchant_Category", "Class_Name"]].head(10))

z_scores = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
z_outliers = z_scores.abs() > 3
print(f"\nAmount outliers by z-score (|z| > 3) : {int(z_outliers.sum())}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(df["Time"], df["Amount"], s=5, color="#4C72B0", alpha=0.4,
                label="inliers")
axes[0].scatter(df.loc[amount_outliers, "Time"], df.loc[amount_outliers, "Amount"],
                s=25, color="crimson", marker="^", label="IQR outliers")
axes[0].set_title("Outliers in Transaction Amount")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Amount")
axes[0].legend()

axes[1].boxplot(df["Amount"], vert=True, showfliers=True,
                flierprops=dict(marker="^", markerfacecolor="crimson",
                                markersize=5, alpha=0.6))
axes[1].set_title("Box plot of Amount with fliers")
axes[1].set_ylabel("Amount")
plt.tight_layout()
save(fig, "lab3_6_outliers.png")

print("\nRelationship between features and the class (point-biserial correlation):")
relations = df.select_dtypes(include=["number"]).corr()["Class"].drop("Class")
relations = relations.reindex(relations.abs().sort_values(ascending=False).index)
print(relations.head(12).round(4))

print("\nAverage Amount by Merchant Category (relationship check):")
print(df.groupby("Merchant_Category")["Amount"].agg(["mean", "median", "count"]).round(2))

print("\n" + "=" * 78)
print("END OF LAB 3")
print("=" * 78)
