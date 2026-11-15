"""
Lab 2: Data Preprocessing and Exploration
Topics:
  Loading datasets
  Handling missing values
  Removing duplicates
  Encoding categorical variables
  Feature scaling (normalization and standardization)
  Descriptive statistics
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # save figures to file (remove this line to display them)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)

# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "credit_card_transactions.csv"


def section(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


print("=" * 78)
print("LAB 2 : DATA PREPROCESSING AND EXPLORATION")
print("=" * 78)

# ==================================================================================
# 1. LOADING DATASETS
# ==================================================================================
section("1. LOADING DATASETS")

df = pd.read_csv(DATA_FILE)
print(f"File loaded from        : {DATA_FILE}")
print(f"Shape (rows, columns)  : {df.shape}")
print(f"Total memory usage      : {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 3 rows:")
print(df.tail(3))

print("\nColumn names:")
print(list(df.columns))

print("\nData types:")
print(df.dtypes)

print("\ninfo():")
df.info()

print("\nShape of each column (non-null values):")
print(df.count())


# ==================================================================================
# 2. HANDLING MISSING VALUES
# ==================================================================================
section("2. HANDLING MISSING VALUES")

print("Numeric columns are filled with the median, categorical columns with the mode,")
print("rows with too many missing values are removed.")

missing_counts = df.isnull().sum()
missing_percent = df.isnull().mean() * 100
missing_report = pd.DataFrame({
    "Missing_Values": missing_counts,
    "Missing_%": missing_percent.round(3),
})
print("Missing value report:")
print(missing_report[missing_report["Missing_Values"] > 0])

total_missing = int(df.isnull().sum().sum())
print(f"\nTotal missing cells     : {total_missing}")
print(f"Total rows with a missing value : {int(df.isnull().any(axis=1).sum())}")

if total_missing > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    cols_with_missing = missing_report[missing_report["Missing_Values"] > 0]
    ax.bar(cols_with_missing.index, cols_with_missing["Missing_Values"], color="crimson")
    ax.set_title("Missing values per column")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "lab2_missing_values.png", dpi=120)
    plt.close(fig)
    print(f"Missing value chart saved -> {OUT_DIR / 'lab2_missing_values.png'}")

df = df.dropna(thresh=df.shape[1] - 2, how="any")
print(f"\nRows after dropping rows with more than 2 missing values : {df.shape[0]}")

numeric_cols = df.select_dtypes(include=["number"]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"  Filled numeric column '{col}' with median {median_value:.4f}")

categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if df[col].isnull().any():
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)
        print(f"  Filled categorical column '{col}' with mode '{mode_value}'")

print(f"\nMissing values after treatment : {int(df.isnull().sum().sum())}")


# ==================================================================================
# 3. REMOVING DUPLICATES
# ==================================================================================
section("3. REMOVING DUPLICATES")

duplicates = int(df.duplicated().sum())
print(f"Number of duplicated rows : {duplicates}")

if duplicates:
    print("\nExample of a duplicated row:")
    print(df[df.duplicated(keep=False)].head())

    df = df.drop_duplicates()
    print(f"\nRows after removing duplicates : {df.shape[0]}")
    print(f"Duplicates remaining           : {int(df.duplicated().sum())}")

subset_dupes = int(df.duplicated(subset=["Time", "Amount", "Merchant_Category"]).sum())
print(f"Duplicates on (Time, Amount, Merchant_Category) : {subset_dupes}")


# ==================================================================================
# 4. ENCODING CATEGORICAL VARIABLES
# ==================================================================================
section("4. ENCODING CATEGORICAL VARIABLES")

print("LabelEncoder -> one integer per category (ordinal / few categories)")
print("OneHotEncoder / get_dummies -> one binary column per category (nominal)")

print("Categorical columns found:", list(categorical_cols))

for col in categorical_cols:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts())
    print(f"Number of unique values: {df[col].nunique()}")

label_encoder = LabelEncoder()
df["Card_Type_Encoded"] = label_encoder.fit_transform(df["Card_Type"])
print(f"\nLabelEncoder classes for Card_Type : {list(label_encoder.classes_)}")
print(df[["Card_Type", "Card_Type_Encoded"]].head())

merchant_dummies = pd.get_dummies(df["Merchant_Category"], prefix="Merchant", dtype=int)
country_dummies = pd.get_dummies(df["Country"], prefix="Country", dtype=int)

print("\nOne-hot encoded Merchant_Category columns:")
print(list(merchant_dummies.columns))
print(merchant_dummies.head())

print("\nOne-hot encoded Country columns:")
print(list(country_dummies.columns))

try:
    ohe = OneHotEncoder(sparse_output=False, drop=None)
except TypeError:
    ohe = OneHotEncoder(sparse=False, drop=None)
arr = ohe.fit_transform(df[["Merchant_Category"]])
print(f"\nOneHotEncoder output shape : {arr.shape}")
print(f"Categories                  : {ohe.categories_[0].tolist()}")

df = pd.concat([df, merchant_dummies, country_dummies], axis=1)
print(f"\nShape after encoding : {df.shape}")


# ==================================================================================
# 5. FEATURE SCALING (NORMALIZATION AND STANDARDIZATION)
# ==================================================================================
section("5. FEATURE SCALING (NORMALIZATION AND STANDARDIZATION)")

print("MinMaxScaler  -> every value is mapped between 0 and 1")
print("StandardScaler -> mean = 0 and standard deviation = 1")
print("Needed by the distance based algorithms (K-Means, SVM, PCA).")

print("Scaling is needed for distance based algorithms (K-Means, SVM, PCA) and for")
print("algorithms that use gradient descent (Logistic Regression, Neural Networks).")

scale_cols = ["Amount", "Time", "V14", "V17"]
features = df[scale_cols].copy()

print("\nBEFORE scaling:")
print(features.describe().round(3))

minmax_scaler = MinMaxScaler()
df[[f"{c}_norm" for c in scale_cols]] = minmax_scaler.fit_transform(features)
print("\nAFTER Min-Max normalization (values between 0 and 1):")
print(df[[f"{c}_norm" for c in scale_cols]].describe().round(3))

standard_scaler = StandardScaler()
df[[f"{c}_std" for c in scale_cols]] = standard_scaler.fit_transform(features)
print("\nAFTER standardization (mean ~ 0, std ~ 1):")
print(df[[f"{c}_std" for c in scale_cols]].describe().round(3))

print("\nSingle row example (scaled values):")
print(df[["Amount", "Amount_norm", "Amount_std"]].head())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(features["Amount"], bins=60, color="steelblue", edgecolor="black")
axes[0].set_title("Original Amount distribution")
axes[1].hist(df["Amount_std"], bins=60, color="darkorange", edgecolor="black")
axes[1].set_title("Standardized Amount distribution")
plt.tight_layout()
fig.savefig(OUT_DIR / "lab2_scaling.png", dpi=120)
plt.close(fig)
print(f"\nScaling comparison chart saved -> {OUT_DIR / 'lab2_scaling.png'}")


# ==================================================================================
# 6. DESCRIPTIVE STATISTICS
# ==================================================================================
section("6. DESCRIPTIVE STATISTICS")

print("describe() for the whole (numeric) dataset:")
print(df.describe().round(3))

print("\nDescriptive statistics of the important columns:")
summary = df[["Time", "Amount", "V12", "V14", "V17"]].describe().T
summary["skew"] = df[["Time", "Amount", "V12", "V14", "V17"]].skew()
summary["kurtosis"] = df[["Time", "Amount", "V12", "V14", "V17"]].kurtosis()
print(summary.round(3))

print("\nMode / median / range of Amount:")
print(f"  mode   : {df['Amount'].mode().iloc[0]:.2f}")
print(f"  median : {df['Amount'].median():.2f}")
print(f"  range  : {df['Amount'].min():.2f}  ->  {df['Amount'].max():.2f}")
print(f"  IQR    : {df['Amount'].quantile(0.75) - df['Amount'].quantile(0.25):.2f}")

print("\nValue counts of the target variable 'Class' (0 = Legitimate, 1 = Fraud):")
print(df["Class"].value_counts())
print(f"Fraud percentage : {df['Class'].mean() * 100:.3f}%")

print("\nGroup-wise statistics (legitimate vs fraud):")
print(df.groupby("Class")[["Amount", "V12", "V14", "V17", "Time"]].mean().round(3))

print("\nTop 10 features correlated with the target:")
corr_with_class = df.select_dtypes(include=["number"]).corr()["Class"].drop("Class")
print(corr_with_class.abs().sort_values(ascending=False).head(10).round(4))

# --- save the preprocessed dataset ---
clean_file = OUT_DIR / "lab2_cleaned_data.csv"
df.to_csv(clean_file, index=False)
print(f"\nPreprocessed dataset written to : {clean_file}")
print(f"Shape of the preprocessed dataset: {df.shape}")

print("\n" + "=" * 78)
print("END OF LAB 2")
print("=" * 78)
