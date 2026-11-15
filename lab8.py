"""
Lab 8: Classification - Support Vector Machines (SVM)
Topics:
  Linear and non linear SVM
  Kernel functions
  Model training
  Hyper parameter tuning (GridSearchCV)
  Accuracy, precision, recall and F1-score
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
print("LAB 8 : CLASSIFICATION - SUPPORT VECTOR MACHINES")
print("=" * 78)


# ==================================================================================
# 1. LINEAR AND NON LINEAR SVM + KERNEL FUNCTIONS
# ==================================================================================
section("1. LINEAR AND NON LINEAR SVM, KERNEL FUNCTIONS")

print("The SVM finds the hyper-plane with the largest margin:")
print("  min 0.5*||w||^2 + C*sum(xi_i)   subject to   y_i (w.x_i + b) >= 1 - xi_i")
print("  C = penalty of the misclassified points, gamma = radius of influence of a point")
print("  linear : K(x,x') = x.x'                -> straight boundary")
print("  poly   : K(x,x') = (gamma*x.x' + r)^d  -> polynomial boundary")
print("  rbf    : K(x,x') = exp(-gamma*||x-x'||^2) -> flexible curved boundary")
print("  sigmoid: K(x,x') = tanh(gamma*x.x' + r)")


# ==================================================================================
# 2. PREPARING THE DATA
# ==================================================================================
section("2. PREPARING THE DATA")

df = pd.read_csv(DATA_FILE).drop_duplicates().dropna().reset_index(drop=True)
categorical = ["Merchant_Category", "Card_Type", "Country"]
df = pd.concat([df.drop(columns=categorical),
                pd.get_dummies(df[categorical], dtype=int)], axis=1)

feature_cols = [c for c in df.columns if c != "Class"]

legit = df[df["Class"] == 0].sample(n=4000, random_state=42)
fraud = df[df["Class"] == 1]
sub = pd.concat([legit, fraud]).sample(frac=1.0, random_state=42).reset_index(drop=True)

X = sub[feature_cols]
y = sub["Class"]
print(f"Sub sample shape : {sub.shape}")
print(f"Class distribution :\n{y.value_counts().to_string()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Training set : {X_train_scaled.shape}   Test set : {X_test_scaled.shape}")


# ==================================================================================
# 3. MODEL TRAINING WITH DIFFERENT KERNELS
# ==================================================================================
section("3. MODEL TRAINING WITH DIFFERENT KERNELS")

kernels = ["linear", "rbf", "poly", "sigmoid"]
kernel_results = []

for kernel in kernels:
    svc = SVC(kernel=kernel, C=1.0, gamma="scale", random_state=42)
    svc.fit(X_train_scaled, y_train)
    pred = svc.predict(X_test_scaled)
    score = accuracy_score(y_test, pred)
    kernel_results.append({
        "kernel": kernel,
        "accuracy": score,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "n_support": int(svc.n_support_.sum()),
        "support_vectors": int(svc.support_vectors_.shape[0]),
    })
    print(f"  kernel = {kernel:8} -> accuracy = {score:.4f}   "
          f"support vectors = {svc.support_vectors_.shape[0]}")

kernel_df = pd.DataFrame(kernel_results)
print("\nComparison of the four kernels:")
print(kernel_df.round(4).to_string(index=False))


# ==================================================================================
# 4. HYPER PARAMETER TUNING WITH GridSearchCV
# ==================================================================================
section("4. HYPER PARAMETER TUNING (GridSearchCV)")

param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(SVC(random_state=42), param_grid=param_grid, cv=cv,
                     scoring="f1", n_jobs=-1, verbose=0)
grid.fit(X_train_scaled, y_train)

print(f"\nGrid searched {len(grid.cv_results_['mean_test_score'])} combinations")
print(f"Best parameters : {grid.best_params_}")
print(f"Best CV F1 score : {grid.best_score_:.4f}")

results_grid = pd.DataFrame({
    "kernel": grid.cv_results_["param_kernel"],
    "C": grid.cv_results_["param_C"],
    "gamma": [str(g) for g in grid.cv_results_["param_gamma"]],
    "mean_F1": grid.cv_results_["mean_test_score"],
    "std_F1": grid.cv_results_["std_test_score"],
}).sort_values("mean_F1", ascending=False)

print("\nTop 10 parameter combinations:")
print(results_grid.head(10).round(4).to_string(index=False))
results_grid.to_csv(OUT_DIR / "lab8_grid_search.csv", index=False)

print("\nEffect of C (regularisation) on a linear SVM:")
for C in [0.01, 0.1, 1, 10, 100]:
    model = SVC(kernel="linear", C=C, random_state=42)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    print(f"  C = {C:<6} -> accuracy = {accuracy_score(y_test, pred):.4f}, "
          f"F1 = {f1_score(y_test, pred, zero_division=0):.4f}")


# ==================================================================================
# 5. EVALUATION OF THE BEST MODEL
# ==================================================================================
section("5. EVALUATION OF THE BEST MODEL")

best_svc = grid.best_estimator_
y_pred = best_svc.predict(X_test_scaled)
scores = best_svc.decision_function(X_test_scaled)

print(f"Best model : {best_svc}")
print("\nPerformance on the test set:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1 score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  ROC AUC   : {roc_auc_score(y_test, scores):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

cv_scores = cross_val_score(best_svc, X_train_scaled, y_train, cv=cv, scoring="accuracy")
print(f"\n3 fold cross validation accuracy : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

fpr, tpr, _ = roc_curve(y_test, scores)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"SVM (AUC = {roc_auc_score(y_test, scores):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Tuned SVM")
ax.legend()
plt.tight_layout()
save(fig, "lab8_roc_curve.png")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Tuned SVM")
plt.tight_layout()
save(fig, "lab8_confusion_matrix.png")


# ==================================================================================
# 6. DECISION BOUNDARIES FOR THE DIFFERENT KERNELS
# ==================================================================================
section("6. DECISION BOUNDARIES FOR THE DIFFERENT KERNELS")

two_features = ["V14", "V17"]
X2 = sub[two_features]
y2 = sub["Class"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3,
                                                        random_state=42, stratify=y2)
sc2 = StandardScaler()
X2_train_s = sc2.fit_transform(X2_train)
X2_test_s = sc2.transform(X2_test)

xx, yy = np.meshgrid(np.linspace(X2_test_s[:, 0].min() - 1, X2_test_s[:, 0].max() + 1, 400),
                     np.linspace(X2_test_s[:, 1].min() - 1, X2_test_s[:, 1].max() + 1, 400))

fig, axes = plt.subplots(1, 4, figsize=(21, 5))
for ax, kernel in zip(axes, ["linear", "rbf", "poly", "sigmoid"]):
    model = SVC(kernel=kernel, C=1.0, gamma="scale", random_state=42)
    model.fit(X2_train_s, y2_train)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm", levels=2)
    ax.scatter(X2_test_s[:, 0], X2_test_s[:, 1], c=y2_test, cmap="coolwarm",
               edgecolor="k", s=22, alpha=0.85)
    ax.set_title(f"{kernel.upper()} kernel")
    ax.set_xlabel("V14 (standardised)")
    ax.set_ylabel("V17 (standardised)")
plt.tight_layout()
save(fig, "lab8_decision_boundaries.png")

print("Figures with the decision boundary of every kernel were saved in outputs/.")

print("\n" + "=" * 78)
print("END OF LAB 8")
print("=" * 78)
