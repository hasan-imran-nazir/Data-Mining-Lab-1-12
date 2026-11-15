"""
Lab 6: Classification - Decision Trees
Topics:
  Decision tree concepts
  Entropy and Gini index (calculated by hand)
  Training a DecisionTreeClassifier
  Tree visualization
  Accuracy evaluation
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
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score, train_test_split)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

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
print("LAB 6 : CLASSIFICATION - DECISION TREES")
print("=" * 78)


# ==================================================================================
# 1. DECISION TREE CONCEPTS
# ==================================================================================
section("1. DECISION TREE CONCEPTS")

print("A decision tree splits the data with questions of the form 'feature <= value ?'.")
print("The split that gives the largest reduction of the impurity is chosen.")
print("  Entropy(S) = - sum( p_i * log2(p_i) )")
print("  Gini(S)    = 1 - sum( p_i^2 )")
print("  Information Gain(S, A) = Entropy(S) - sum( |S_v|/|S| * Entropy(S_v) )")
print("  Gini gain(S, A)        = Gini(S) - sum( |S_v|/|S| * Gini(S_v) )")


# ==================================================================================
# 2. ENTROPY AND GINI INDEX - CALCULATED BY HAND
# ==================================================================================
section("2. ENTROPY AND GINI INDEX (MANUAL CALCULATION)")


def entropy(labels):
    """Shannon entropy of a label list."""
    labels = list(labels)
    if not labels:
        return 0.0
    total = len(labels)
    ent = 0.0
    for count in [labels.count(c) for c in set(labels)]:
        p = count / total
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def gini(labels):
    """Gini impurity of a label list."""
    labels = list(labels)
    if not labels:
        return 0.0
    total = len(labels)
    return 1 - sum((labels.count(c) / total) ** 2 for c in set(labels))


demo = pd.DataFrame({
    "Amount": [10, 20, 30, 40, 50, 60],
    "Class":  [0, 0, 0, 1, 1, 1],
})
print("Training data:")
print(demo.to_string(index=False))

parent_entropy = entropy(demo["Class"])
parent_gini = gini(demo["Class"])
print(f"\nParent node  -> Entropy = {parent_entropy:.4f}   Gini = {parent_gini:.4f}")

left = demo[demo["Amount"] <= 30]["Class"]
right = demo[demo["Amount"] > 30]["Class"]
child_entropy = (len(left) / len(demo)) * entropy(left) + (len(right) / len(demo)) * entropy(right)
child_gini = (len(left) / len(demo)) * gini(left) + (len(right) / len(demo)) * gini(right)

print(f"  Left  (Amount <= 30): n={len(left)} labels={list(left)} "
      f"Entropy={entropy(left):.4f} Gini={gini(left):.4f}")
print(f"  Right (Amount >  30): n={len(right)} labels={list(right)} "
      f"Entropy={entropy(right):.4f} Gini={gini(right):.4f}")
print(f"\nWeighted entropy of the children : {child_entropy:.4f}")
print(f"Information gain (entropy)        : {parent_entropy - child_entropy:.4f}")
print(f"Weighted gini of the children     : {child_gini:.4f}")
print(f"Gini gain                         : {parent_gini - child_gini:.4f}")
print("=> The split creates two PURE leaves, so the information gain is maximum.")

print("\nEntropy values for different class ratios (0 = pure, 1 = maximum):")
for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    labels = [0] * int(p * 100) + [1] * (100 - int(p * 100))
    print(f"  p(class0) = {p:.2f}  ->  Entropy = {entropy(labels):.4f}   "
          f"Gini = {gini(labels):.4f}")


# ==================================================================================
# 3. PREPARING THE DATA
# ==================================================================================
section("3. PREPARING THE DATA")

df = pd.read_csv(DATA_FILE).drop_duplicates().reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

categorical = ["Merchant_Category", "Card_Type", "Country"]
df = pd.concat([df.drop(columns=categorical),
                pd.get_dummies(df[categorical], dtype=int)], axis=1)

feature_cols = [c for c in df.columns if c not in ("Class", "Class_Name")]
X = df[feature_cols]
y = df["Class"]

print(f"Dataset shape      : {df.shape}")
print(f"Number of features : {len(feature_cols)}")
print(f"Class distribution :\n{y.value_counts().to_string()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTraining set : {X_train.shape}   Test set : {X_test.shape}")
print(f"Fraud in train: {y_train.sum()}   Fraud in test: {y_test.sum()}")


# ==================================================================================
# 4. TRAINING A DECISION TREE CLASSIFIER
# ==================================================================================
section("4. TRAINING A DECISION TREE CLASSIFIER")

results = {}

for criterion in ["gini", "entropy"]:
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=6, min_samples_leaf=10,
                                  class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[criterion] = (clf, acc)
    print(f"\ncriterion = {criterion}")
    print(f"  Test accuracy  : {acc:.4f}")
    print(f"  Test precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Test recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Test F1 score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Tree depth     : {clf.get_depth()}   leaves: {clf.get_n_leaves()}")

best_criterion, (best_clf, best_acc) = max(results.items(), key=lambda kv: kv[1][1])
print(f"\nBest criterion : {best_criterion} (accuracy {best_acc:.4f})")


# ==================================================================================
# 5. TREE VISUALIZATION
# ==================================================================================
section("5. TREE VISUALIZATION")

tree_text = export_text(best_clf, feature_names=[str(c) for c in feature_cols],
                        decimals=2, max_depth=4)
print("Textual representation of the tree (first 30 lines):")
print("\n".join(tree_text.splitlines()[:30]))

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best_clf, feature_names=[str(c) for c in feature_cols],
          class_names=CLASS_NAMES, filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree (fraud detection)")
plt.tight_layout()
save(fig, "lab6_decision_tree.png")

small_clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
small_clf.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(small_clf, feature_names=[str(c) for c in feature_cols],
          class_names=CLASS_NAMES, filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title("Decision Tree with max_depth = 3")
plt.tight_layout()
save(fig, "lab6_decision_tree_depth3.png")

imp = pd.Series(best_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features used by the tree:")
print(imp.head(10).round(4).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(imp.head(10).index[::-1], imp.head(10).values[::-1], color="#4C72B0", edgecolor="black")
ax.set_xlabel("Feature importance (Gini decrease)")
ax.set_title("Most important features for the decision tree")
plt.tight_layout()
save(fig, "lab6_feature_importance.png")


# ==================================================================================
# 6. ACCURACY EVALUATION
# ==================================================================================
section("6. ACCURACY EVALUATION")

y_pred = best_clf.predict(X_test)
y_prob = best_clf.predict_proba(X_test)[:, 1]

print("Confusion matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

print(f"ROC AUC score : {roc_auc_score(y_test, y_prob):.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_clf, X, y, cv=cv, scoring="accuracy")
print(f"\n5 fold cross validation accuracy : {scores.mean():.4f} (+/- {scores.std():.4f})")
print("  fold scores:", np.round(scores, 4).tolist())

print("\nEffect of max_depth (over-fitting study):")
print(f"{'depth':>6} {'train acc':>10} {'test acc':>10} {'leaves':>8}")
depth_results = []
for depth in range(1, 13):
    tree_clf = DecisionTreeClassifier(max_depth=depth, class_weight="balanced",
                                      random_state=42)
    tree_clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree_clf.predict(X_train))
    test_acc = accuracy_score(y_test, tree_clf.predict(X_test))
    depth_results.append({"max_depth": depth, "train_accuracy": train_acc,
                           "test_accuracy": test_acc, "leaves": tree_clf.get_n_leaves()})
    print(f"{depth:>6} {train_acc:>10.4f} {test_acc:>10.4f} {tree_clf.get_n_leaves():>8}")

depth_df = pd.DataFrame(depth_results)
depth_df.to_csv(OUT_DIR / "lab6_depth_study.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(depth_df["max_depth"], depth_df["train_accuracy"], marker="o", label="Training accuracy")
ax.plot(depth_df["max_depth"], depth_df["test_accuracy"], marker="s", label="Test accuracy")
ax.set_xlabel("Maximum depth of the tree")
ax.set_ylabel("Accuracy")
ax.set_title("Over-fitting: training vs test accuracy")
ax.legend()
plt.tight_layout()
save(fig, "lab6_overfitting.png")

print("\nCost complexity pruning (ccp_alpha):")
for alpha in [0.0, 0.0001, 0.001, 0.01, 0.05]:
    pruned = DecisionTreeClassifier(max_depth=8, ccp_alpha=alpha, class_weight="balanced",
                                    random_state=42)
    pruned.fit(X_train, y_train)
    print(f"  ccp_alpha = {alpha:<7} -> accuracy = "
          f"{accuracy_score(y_test, pruned.predict(X_test)):.4f}, "
          f"leaves = {pruned.get_n_leaves()}")

print("\n" + "=" * 78)
print("END OF LAB 6")
print("=" * 78)
