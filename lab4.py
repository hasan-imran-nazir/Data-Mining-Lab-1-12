"""
Lab 4: Frequent Pattern Mining - Apriori Algorithm
Topics:
  Association rules
  Support
  Confidence
  Lift
  Apriori implementation using Mlxtend
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
DATA_FILE = DATA_DIR / "market_basket_transactions.csv"


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
print("LAB 4 : FREQUENT PATTERN MINING - APRIORI ALGORITHM")
print("=" * 78)


# ==================================================================================
# 1. ASSOCIATION RULES - THE BASIC MEASURES
# ==================================================================================
section("1. ASSOCIATION RULES, SUPPORT, CONFIDENCE AND LIFT")

print("Rule:  X -> Y")
print("  Support(A)       = count(A) / count(all transactions)")
print("  Support(A -> B)  = count(A and B) / count(all transactions)")
print("  Confidence(A->B) = Support(A and B) / Support(A)")
print("  Lift(A -> B)     = Support(A and B) / (Support(A) * Support(B))")
print("  Lift > 1 : A and B occur together more often than expected")


mini_basket = [
    ["Bread", "Butter"],
    ["Bread", "Butter", "Milk"],
    ["Bread", "Milk"],
    ["Butter", "Milk"],
    ["Bread", "Butter", "Milk"],
]
mini = pd.DataFrame(0, index=range(len(mini_basket)),
                    columns=["Bread", "Butter", "Milk"], dtype=int)
for i, tx in enumerate(mini_basket):
    for item in tx:
        mini.loc[i, item] = 1

total = len(mini)
sup_bread = mini["Bread"].sum() / total
sup_butter = mini["Butter"].sum() / total
sup_milk = mini["Milk"].sum() / total
sup_bread_butter = mini[["Bread", "Butter"]].all(axis=1).sum() / total
conf_bread_butter = sup_bread_butter / sup_bread
lift_bread_butter = sup_bread_butter / (sup_bread * sup_butter)

print("Hand worked example (5 transactions: Bread, Butter, Milk)")
print(f"  Support(Bread)              = {sup_bread:.3f}")
print(f"  Support(Butter)             = {sup_butter:.3f}")
print(f"  Support(Milk)               = {sup_milk:.3f}")
print(f"  Support(Bread and Butter)   = {sup_bread_butter:.3f}")
print(f"  Confidence(Bread -> Butter) = {conf_bread_butter:.3f}")
print(f"  Lift(Bread -> Butter)       = {lift_bread_butter:.3f}")
print("  => Lift > 1, buying Bread strongly increases the chance of buying Butter.")


# ==================================================================================
# 2. LOADING AND PREPARING THE TRANSACTION DATA
# ==================================================================================
section("2. LOADING THE TRANSACTION DATA")

df = pd.read_csv(DATA_FILE)
print(f"Dataset : {DATA_FILE}")
print(f"Shape   : {df.shape}")
print(df.head())

baskets = df["Items"].apply(lambda s: [i.strip() for i in str(s).split(",")])
print(f"\nExample transaction : {baskets.iloc[0]}")

all_items = sorted({item for tx in baskets for item in tx})
print(f"\nNumber of transactions : {len(baskets)}")
print(f"Number of distinct items : {len(all_items)}")
print(f"Items : {all_items}")

print("\nItem frequency (support of each single item):")
item_support = baskets.explode().value_counts() / len(baskets)
print((item_support * 100).round(2).rename("support %").to_frame().sort_values(
    "support %", ascending=False))

one_hot = pd.DataFrame(0, index=df.index, columns=all_items, dtype=int)
for idx, tx in baskets.items():
    for item in tx:
        one_hot.loc[idx, item] = 1

print(f"\nOne hot encoded matrix shape : {one_hot.shape}")
print(one_hot.head())

one_hot_bool = one_hot.astype(bool)


# ==================================================================================
# 3. APRIORI - MINING FREQUENT ITEMSETS
# ==================================================================================
section("3. APRIORI - MINING FREQUENT ITEMSETS")


MIN_SUPPORT = 0.10

try:
    frequent_itemsets = apriori(one_hot_bool, min_support=MIN_SUPPORT,
                                 use_interesting_measures=False)
except Exception:                 # fallback for older mlxtend versions
    frequent_itemsets = apriori(one_hot_bool.values, min_support=MIN_SUPPORT,
                                use_interesting_measures=False)

frequent_itemsets = frequent_itemsets.reset_index(drop=True)
frequent_itemsets["n_items"] = frequent_itemsets["itemsets"].apply(len)

print(f"\nFrequent itemsets found with min_support = {MIN_SUPPORT}")
print(f"Total frequent itemsets : {len(frequent_itemsets)}")
print(frequent_itemsets.sort_values(["n_items", "support"], ascending=[True, False])
      .head(20).to_string(index=False))

level_counts = frequent_itemsets.groupby("n_items").size()
print("\nNumber of itemsets per level (k = number of items):")
print(level_counts.rename("itemsets").to_frame())

print("\nEffect of min_support on the number of frequent itemsets:")
for s in [0.05, 0.10, 0.15, 0.20, 0.25]:
    try:
        fi = apriori(one_hot_bool, min_support=s, use_interesting_measures=False)
        print(f"  min_support = {s:.2f}  ->  {len(fi):3d} itemsets")
    except Exception:
        fi = apriori(one_hot_bool.values, min_support=s, use_interesting_measures=False)
        print(f"  min_support = {s:.2f}  ->  {len(fi):3d} itemsets")

frequent_itemsets.to_csv(OUT_DIR / "lab4_frequent_itemsets.csv", index=False)
print(f"\nFrequent itemsets saved -> {OUT_DIR / 'lab4_frequent_itemsets.csv'}")


# ==================================================================================
# 4. GENERATING ASSOCIATION RULES
# ==================================================================================
section("4. GENERATING ASSOCIATION RULES")

MIN_CONFIDENCE = 0.60

rules = association_rules(frequent_itemsets, metric="confidence",
                          min_threshold=MIN_CONFIDENCE)

if rules.empty:
    MIN_CONFIDENCE = 0.40
    rules = association_rules(frequent_itemsets, metric="confidence",
                              min_threshold=MIN_CONFIDENCE)

rules = rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)
rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

print(f"\nTotal rules with min_confidence = {MIN_CONFIDENCE} : {len(rules)}")
print(rules.head(15).to_string(index=False))

interesting = rules[(rules["lift"] > 1) & (rules["confidence"] >= 0.6)]
print(f"\nInteresting rules (lift > 1 and confidence >= 0.6) : {len(interesting)}")
print(interesting[["antecedents", "consequents", "support", "confidence", "lift"]]
      .head(15).to_string(index=False))

if not rules.empty:
    row = rules.iloc[0]
    ante = [a.strip() for a in str(row["antecedents"]).split(",")]
    cons = [c.strip() for c in str(row["consequents"]).split(",")]
    mask_ante = one_hot[ante].all(axis=1)
    mask_cons = one_hot[cons].all(axis=1)
    n_ante = int(mask_ante.sum())
    n_both = int((mask_ante & mask_cons).sum())
    print("\nManual verification of the top rule:")
    print(f"  rule                 : {{{', '.join(ante)}}} -> {{{', '.join(cons)}}}")
    print(f"  support  (calculated) : {n_both / len(one_hot):.4f}   (mlxtend: {row['support']:.4f})")
    print(f"  confidence(calculated) : {n_both / n_ante:.4f}   (mlxtend: {row['confidence']:.4f})")
    print(f"  lift     (calculated) : {(n_both / len(one_hot)) / ((n_ante / len(one_hot)) * (mask_cons.sum() / len(one_hot))):.4f}"
          f"   (mlxtend: {row['lift']:.4f})")

rules.to_csv(OUT_DIR / "lab4_association_rules.csv", index=False)
print(f"\nRules saved -> {OUT_DIR / 'lab4_association_rules.csv'}")


# ==================================================================================
# 5. VISUALISING THE RESULTS
# ==================================================================================
section("5. VISUALISING THE RESULTS")

if not rules.empty:
    top = rules.head(12).copy()
    labels = [f"{a} -> {c}" for a, c in zip(top["antecedents"], top["consequents"])]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(labels[::-1], top["lift"][::-1], color="#C44E52", edgecolor="black")
    ax.set_xlabel("Lift")
    ax.set_title("Top 12 rules by lift")
    plt.tight_layout()
    save(fig, "lab4_top_rules_lift.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sizes = (top["lift"] * 120).clip(lower=40)
    sc = ax.scatter(top["support"], top["confidence"], s=sizes,
                    c=top["lift"], cmap="viridis", alpha=0.75, edgecolor="black")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Support vs Confidence (size/colour = lift)")
    fig.colorbar(sc, ax=ax, label="Lift")
    plt.tight_layout()
    save(fig, "lab4_support_confidence.png")

    rules["rule_length"] = rules["antecedents"].apply(lambda s: len(s.split(","))) + \
        rules["consequents"].apply(lambda s: len(s.split(",")))
    print("\nNumber of rules by rule length:")
    print(rules["rule_length"].value_counts().sort_index().rename("rules").to_frame())

print("\n" + "=" * 78)
print("END OF LAB 4")
print("=" * 78)
