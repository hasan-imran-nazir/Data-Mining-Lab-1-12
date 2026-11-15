"""
Lab 7: Classification - Naive Bayes
Topics:
  Bayes' theorem
  Gaussian Naive Bayes
  Multinomial Naive Bayes
  Classification of text data and tabular data
  Performance evaluation
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
TX_FILE = DATA_DIR / "credit_card_transactions.csv"
TEXT_FILE = DATA_DIR / "text_corpus.csv"


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
print("LAB 7 : CLASSIFICATION - NAIVE BAYES")
print("=" * 78)


# ==================================================================================
# 1. BAYES' THEOREM
# ==================================================================================
section("1. BAYES' THEOREM")

print("  P(A | B) = P(B | A) * P(A) / P(B)")
print("  Naive Bayes: P(C | x1..xn)  ~  P(C) * P(x1 | C) * ... * P(xn | C)")
print("  GaussianNB = continuous features, MultinomialNB = word counts, "
      "BernoulliNB = present / absent")


freq_prior = 0.01            # 1 % of the transactions are fraudulent
p_fraud_given_large = 0.30
p_fraud_given_small = 0.02
p_large = 0.10

posterior_fraud = (p_fraud_given_large * freq_prior) / p_large
print("Numerical example:")
print(f"  P(fraud)                    = {freq_prior}")
print(f"  P(large amount | fraud)     = {p_fraud_given_large}")
print(f"  P(large amount | legitimate) = {p_fraud_given_small}")
print(f"  P(large amount)              = {p_large}")
print(f"  P(fraud | large amount)      = {posterior_fraud:.4f}")
print("  => a large transaction is much more likely to be fraudulent.")


# ==================================================================================
# 2. GAUSSIAN NAIVE BAYES - TABULAR DATA
# ==================================================================================
section("2. GAUSSIAN NAIVE BAYES ON TABULAR DATA")

df = pd.read_csv(TX_FILE).drop_duplicates().dropna().reset_index(drop=True)
categorical = ["Merchant_Category", "Card_Type", "Country"]
df = pd.concat([df.drop(columns=categorical),
                pd.get_dummies(df[categorical], dtype=int)], axis=1)

feature_cols = [c for c in df.columns if c != "Class"]
X = df[feature_cols]
y = df["Class"]
CLASS_NAMES = ["Legitimate", "Fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)
print(f"Training set : {X_train.shape}   Test set : {X_test.shape}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gnb", GaussianNB()),
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nGaussian Naive Bayes results:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1 score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  ROC AUC   : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("\nLearned parameters of the Gaussian model (first 5 features):")
param_df = pd.DataFrame({
    "feature": feature_cols[:5],
    "mean_legitimate": gnb.theta_[0][:5],
    "mean_fraud": gnb.theta_[1][:5],
    "var_legitimate": gnb.var_[:, 0][:5],
    "var_fraud": gnb.var_[:, 1][:5],
})
print(param_df.round(4).to_string(index=False))

sample = X_test.head(3)
manual_probs = []
for _, row in sample.iterrows():
    log_prior = np.log(gnb.class_prior_)
    log_lik = np.zeros(2)
    for cls in range(2):
        total = 0.0
        for j, feature in enumerate(feature_cols):
            mu = gnb.theta_[cls][j]
            sigma2 = gnb.var_[j][cls]
            total += -0.5 * np.log(2 * np.pi * sigma2) - ((row[feature] - mu) ** 2) / (2 * sigma2)
        log_lik[cls] = total
    post = np.exp(log_prior + log_lik)
    post = post / post.sum()
    manual_probs.append(post[1])

print("\nManual vs model probability for three test transactions:")
comparison = pd.DataFrame({
    "model_probability": gnb.predict_proba(sample)[:, 1],
    "manual_probability": manual_probs,
})
comparison["difference"] = (comparison["model_probability"] - comparison["manual_probability"]).abs()
print(comparison.round(6).to_string())


# ==================================================================================
# 3. MULTINOMIAL NAIVE BAYES - TEXT DATA
# ==================================================================================
section("3. MULTINOMIAL NAIVE BAYES ON TEXT DATA")

text_df = pd.read_csv(TEXT_FILE)
print(f"Text corpus : {text_df.shape}")
print(text_df.head(3).to_string(index=False))
print("\nLabel distribution:")
print(text_df["Label"].value_counts().to_string())

X_text = text_df["Text"]
y_text = text_df["Label"]

X_tr, X_te, y_tr, y_te = train_test_split(X_text, y_text, test_size=0.30,
                                            random_state=42, stratify=y_text)

vectorizer = CountVectorizer(stop_words="english", lowercase=True)
X_tr_counts = vectorizer.fit_transform(X_tr)
X_te_counts = vectorizer.transform(X_te)
vocabulary = vectorizer.get_feature_names_out()

print(f"\nVocabulary size : {len(vocabulary)}")
print(f"Most frequent words : {', '.join(vocabulary[:15])}")

mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_tr_counts, y_tr)
text_pred = mnb.predict(X_te)

print("\nMultinomial Naive Bayes (text) results:")
print(f"  Accuracy  : {accuracy_score(y_te, text_pred):.4f}")
print(f"  Precision : {precision_score(y_te, text_pred, average='macro', zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_te, text_pred, average='macro', zero_division=0):.4f}")
print(f"  F1 score  : {f1_score(y_te, text_pred, average='macro', zero_division=0):.4f}")

print("\nClassification report:")
print(classification_report(y_te, text_pred, zero_division=0))

print("\nConfusion matrix (rows = actual, columns = predicted):")
labels = sorted(y_te.unique())
print(pd.DataFrame(confusion_matrix(y_te, text_pred, labels=labels),
                   index=labels, columns=labels).to_string())

print("\nTop 10 words for each class (log probability):")
for cls in mnb.classes_:
    idx = list(mnb.classes_).index(cls)
    top_idx = np.argsort(mnb.feature_log_prob_[idx])[-10:][::-1]
    words = ", ".join(vocabulary[i] for i in top_idx)
    print(f"  {cls:10}: {words}")

print("\nPredictions for new documents:")
new_docs = [
    "The team scored a goal in the last minute of the football match",
    "The senate approved the new bill after a long debate in parliament",
]
new_matrix = vectorizer.transform(new_docs)
for doc, proba in zip(new_docs, mnb.predict_proba(new_matrix)):
    top = mnb.classes_[int(np.argmax(proba))]
    print(f"  '{doc[:55]}...' -> {top:10} (P = {max(proba):.3f})")

bnb = BernoulliNB()
bnb.fit(X_tr_counts > 0, y_tr)
bnb_pred = bnb.predict(X_te_counts > 0)
print(f"\nBernoulliNB accuracy on the same data : {accuracy_score(y_te, bnb_pred):.4f}")


# ==================================================================================
# 4. PERFORMANCE EVALUATION AND COMPARISON
# ==================================================================================
section("4. PERFORMANCE EVALUATION AND COMPARISON")

models = {
    "GaussianNB (tabular)": (pipeline, X_test, y_test, "predict_proba"),
    "MultinomialNB (text)": (mnb, X_te_counts, y_te, "predict_proba"),
    "BernoulliNB (text)": (bnb, X_te_counts, y_te, "predict_proba"),
}

print(f"{'Model':>24} | {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}")
print("-" * 68)
for name, (model, Xt, yt, prob_method) in models.items():
    pred = model.predict(Xt)
    avg = "macro" if "text" in name else "binary"
    print(f"{name:>24} | {accuracy_score(yt, pred):>9.4f} "
          f"{precision_score(yt, pred, average=avg, zero_division=0):>10.4f} "
          f"{recall_score(yt, pred, average=avg, zero_division=0):>8.4f} "
          f"{f1_score(yt, pred, average=avg, zero_division=0):>7.4f}")

from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(7, 5))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"GaussianNB (AUC = {roc_auc_score(y_test, y_prob):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Gaussian Naive Bayes")
ax.legend()
plt.tight_layout()
save(fig, "lab7_roc_curve.png")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Gaussian Naive Bayes")
plt.tight_layout()
save(fig, "lab7_confusion_matrix.png")

print("\n" + "=" * 78)
print("END OF LAB 7")
print("=" * 78)
