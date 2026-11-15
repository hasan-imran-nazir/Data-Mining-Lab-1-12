"""
Lab 5: Frequent Pattern Mining - FP-Growth
Topics:
  FP-Tree construction (implemented from scratch below)
  FP-Growth algorithm
  Mining frequent itemsets
  Comparison between Apriori and FP-Growth
Run:  python <this file>.py
"""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

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
print("LAB 5 : FREQUENT PATTERN MINING - FP-GROWTH")
print("=" * 78)


# ==================================================================================
# 1. FP-TREE CONSTRUCTION (implemented from scratch)
# ==================================================================================
section("1. FP-TREE CONSTRUCTION")

print("FP-Growth never generates candidate itemsets, that is why it is faster than Apriori.")
print("  Step 1 : count the support of every item (one scan of the database)")
print("  Step 2 : remove the items with support < min_support and sort the rest by")
print("           decreasing support")
print("  Step 3 : insert every sorted transaction into the FP-Tree, the common prefix")
print("           is shared and every node keeps the count of passing transactions")
print("  Step 4 : take the least frequent item, build its conditional pattern base")
print("           and a conditional FP-tree, then mine that tree recursively")


class FPNode:
    """A single node of the FP-Tree."""

    __slots__ = ("item", "count", "parent", "children")

    def __init__(self, item, parent=None):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}


class FPTree:
    """A minimal FP-Tree implementation used for demonstration purposes."""

    def __init__(self, transactions, min_support=0.1):
        self.min_support = min_support
        self.total = len(transactions)

        # Step 1 - count the support of every item
        counts = {}
        for tx in transactions:
            for item in set(tx):
                counts[item] = counts.get(item, 0) + 1
        self.item_counts = counts
        self.support = {item: c / self.total for item, c in counts.items()}

        self.order = {item: idx for idx, item in
                      enumerate(sorted(counts, key=lambda i: (-counts[i], i)))}

        # Step 2 - remove infrequent items and sort each transaction
        self.filtered = [[i for i in set(tx) if self.support.get(i, 0) >= min_support]
                         for tx in transactions]

        # Step 3 - build the tree
        self.root = FPNode(None)
        for tx in self.filtered:
            if tx:
                self.insert(sorted(tx, key=lambda i: self.order[i]))

    def insert(self, items):
        """Insert one sorted transaction into the tree."""
        node = self.root
        for item in items:
            if item not in node.children:
                node.children[item] = FPNode(item, node)
            node = node.children[item]
            node.count += 1

    def display(self, node=None, indent=0):
        """Print the tree in an indented text form."""
        node = self.root if node is None else node
        for item, child in sorted(node.children.items(), key=lambda kv: (-kv[1].count, kv[0])):
            print("    " * indent + f"- {item} (count = {child.count}, "
                                     f"support = {child.count / self.total:.3f})")
            self.display(child, indent + 1)

    def pattern_base(self, item):
        """Conditional pattern base of `item`: list of (path, count) tuples."""
        base = []

        def walk(node, path):
            for child_item, child in node.children.items():
                if child_item == item:
                    base.append((list(path), child.count))
                else:
                    walk(child, path + [child_item])

        walk(self.root, [])
        return base


df = pd.read_csv(DATA_FILE)
baskets = df["Items"].apply(lambda s: [i.strip() for i in str(s).split(",")])
all_items = sorted({item for tx in baskets for item in tx})
one_hot = pd.DataFrame(0, index=df.index, columns=all_items, dtype=int)
for idx, tx in baskets.items():
    for item in tx:
        one_hot.loc[idx, item] = 1
one_hot_bool = one_hot.astype(bool)

MIN_SUPPORT = 0.10
print(f"Transactions : {len(baskets)}   distinct items : {len(all_items)}   "
      f"min_support : {MIN_SUPPORT}")

tree = FPTree(baskets, min_support=MIN_SUPPORT)
print("\nFP-TREE STRUCTURE (root -> leaves):")
tree.display()

rare_items = sorted(tree.support.items(), key=lambda kv: kv[1])
for item, sup in rare_items[:2]:
    base = tree.pattern_base(item)
    print(f"\nConditional pattern base of '{item}' (support {sup:.3f}):")
    for path, count in base:
        print(f"    path = {path}   count = {count}")


# ==================================================================================
# 2. FP-GROWTH - MINING FREQUENT ITEMSETS
# ==================================================================================
section("2. FP-GROWTH - MINING FREQUENT ITEMSETS")

try:
    fp_itemsets = fpgrowth(one_hot_bool, min_support=MIN_SUPPORT,
                           use_interesting_measures=False)
except Exception:
    fp_itemsets = fpgrowth(one_hot_bool.values, min_support=MIN_SUPPORT,
                           use_interesting_measures=False)

fp_itemsets = fp_itemsets.reset_index(drop=True)
fp_itemsets["n_items"] = fp_itemsets["itemsets"].apply(len)
fp_itemsets = fp_itemsets.sort_values(["n_items", "support"], ascending=[True, False])

print(f"\nFrequent itemsets found with FP-Growth (min_support = {MIN_SUPPORT}): {len(fp_itemsets)}")
print(fp_itemsets.head(20).to_string(index=False))

print("\nNumber of itemsets per level:")
print(fp_itemsets.groupby("n_items").size().rename("itemsets").to_frame())

fp_itemsets.to_csv(OUT_DIR / "lab5_fp_frequent_itemsets.csv", index=False)
print(f"\nFrequent itemsets saved -> {OUT_DIR / 'lab5_fp_frequent_itemsets.csv'}")

rules = association_rules(fp_itemsets, metric="confidence", min_threshold=0.60)
if rules.empty:
    rules = association_rules(fp_itemsets, metric="confidence", min_threshold=0.40)
rules = rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)
rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

print(f"\nAssociation rules generated from FP-Growth : {len(rules)}")
print(rules.head(15).to_string(index=False))
rules.to_csv(OUT_DIR / "lab5_fp_association_rules.csv", index=False)


# ==================================================================================
# 3. COMPARISON BETWEEN APRIORI AND FP-GROWTH
# ==================================================================================
section("3. COMPARISON BETWEEN APRIORI AND FP-GROWTH")

print(f"{'min_support':>12} | {'Apriori items':>14} {'Apriori time':>13} | "
      f"{'FP items':>10} {'FP time':>10} | {'Same?':>6}")
print("-" * 78)

comparison = []
for support in [0.05, 0.10, 0.15, 0.20, 0.25]:
    start = time.perf_counter()
    try:
        ap_set = apriori(one_hot_bool, min_support=support, use_interesting_measures=False)
    except Exception:
        ap_set = apriori(one_hot_bool.values, min_support=support,
                         use_interesting_measures=False)
    ap_time = time.perf_counter() - start

    start = time.perf_counter()
    try:
        fp_set = fpgrowth(one_hot_bool, min_support=support, use_interesting_measures=False)
    except Exception:
        fp_set = fpgrowth(one_hot_bool.values, min_support=support,
                          use_interesting_measures=False)
    fp_time = time.perf_counter() - start

    same = set(ap_set["itemsets"]) == set(fp_set["itemsets"])
    comparison.append({
        "min_support": support,
        "apriori_itemsets": len(ap_set),
        "apriori_seconds": round(ap_time, 4),
        "fpgrowth_itemsets": len(fp_set),
        "fpgrowth_seconds": round(fp_time, 4),
        "identical_itemsets": same,
    })
    print(f"{support:>12.2f} | {len(ap_set):>14} {ap_time:>12.4f}s | "
          f"{len(fp_set):>10} {fp_time:>9.4f}s | {str(same):>6}")

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv(OUT_DIR / "lab5_apriori_vs_fpgrowth.csv", index=False)

print("\nInterpretation:")
print("  * Both algorithms return exactly the same frequent itemsets.")
print("  * FP-Growth never generates candidate itemsets, so it is much faster on")
print("    large databases and needs far less memory than Apriori.")
print("  * Apriori is simpler to implement, FP-Growth is better for big data.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(comparison_df["min_support"], comparison_df["apriori_itemsets"],
             marker="o", label="Apriori", color="#C44E52")
axes[0].plot(comparison_df["min_support"], comparison_df["fpgrowth_itemsets"],
             marker="s", label="FP-Growth", color="#4C72B0", linestyle="--")
axes[0].set_xlabel("min_support")
axes[0].set_ylabel("Number of frequent itemsets")
axes[0].set_title("Itemsets found by both algorithms")
axes[0].legend()

axes[1].plot(comparison_df["min_support"], comparison_df["apriori_seconds"],
             marker="o", label="Apriori", color="#C44E52")
axes[1].plot(comparison_df["min_support"], comparison_df["fpgrowth_seconds"],
             marker="s", label="FP-Growth", color="#4C72B0", linestyle="--")
axes[1].set_xlabel("min_support")
axes[1].set_ylabel("Execution time (seconds)")
axes[1].set_title("Execution time comparison")
axes[1].legend()
plt.tight_layout()
save(fig, "lab5_apriori_vs_fpgrowth.png")

print("\n" + "=" * 78)
print("END OF LAB 5")
print("=" * 78)
