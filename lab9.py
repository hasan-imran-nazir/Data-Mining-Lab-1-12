"""
Lab 9: Clustering - K-Means
Topics:
  Distance measures
  K-Means algorithm (implemented by hand and with scikit-learn)
  Choosing the number of clusters (elbow method + silhouette)
  Cluster visualisation
  Interpretation of the clusters
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
DATA_FILE = DATA_DIR / "customer_segments.csv"


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
print("LAB 9 : CLUSTERING - K-MEANS")
print("=" * 78)


# ==================================================================================
# 1. DISTANCE MEASURES
# ==================================================================================
section("1. DISTANCE MEASURES")

print("  Euclidean : sqrt( sum( (a - b)^2 ) )      L2, straight line")
print("  Manhattan : sum( |a - b| )                 L1, city block")
print("  Minkowski : ( sum( |a - b|^p ) ) ^ (1/p)   general form")
print("  Chebyshev : max( |a - b| )                L-infinity")
print("  Cosine    : 1 - (a.b) / (||a|| * ||b||)   angle between the vectors")


def euclidean(a, b):
    """Straight line distance."""
    return float(np.sqrt(np.sum((np.asarray(a) - np.asarray(b)) ** 2)))


def manhattan(a, b):
    """City block distance (L1)."""
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def minkowski(a, b, p=3):
    """Generalised distance, p = 2 is Euclidean, p = 1 is Manhattan."""
    diff = np.abs(np.asarray(a) - np.asarray(b))
    return float(np.sum(diff ** p) ** (1 / p))


def chebyshev(a, b):
    """Maximum absolute difference (L-infinity)."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def cosine_similarity(a, b):
    """Cosine similarity of two vectors."""
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)


point_a = [1.0, 2.0, 3.0]
point_b = [4.0, 6.0, 3.0]
print("Comparison of the distance measures for the points")
print(f"  A = {point_a}")
print(f"  B = {point_b}")
print(f"  Euclidean distance        : {euclidean(point_a, point_b):.4f}")
print(f"  Manhattan distance        : {manhattan(point_a, point_b):.4f}")
print(f"  Minkowski distance (p=3)  : {minkowski(point_a, point_b):.4f}")
print(f"  Chebyshev distance        : {chebyshev(point_a, point_b):.4f}")
print(f"  Cosine similarity         : {cosine_similarity(point_a, point_b):.4f}")
print(f"  Cosine distance           : {cosine_distance(point_a, point_b):.4f}")


# ==================================================================================
# 2. LOADING AND PREPARING THE DATA
# ==================================================================================
section("2. LOADING AND PREPARING THE DATA")

df = pd.read_csv(DATA_FILE)
print(f"Dataset : {DATA_FILE}")
print(f"Shape   : {df.shape}")
print(df.head().to_string(index=False))

numeric_cols = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency",
                "Online_Orders", "Avg_Basket_Size", "Recency_Days"]
X_raw = df[numeric_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print("\nFeatures used for clustering:")
for c in numeric_cols:
    print(f"  {c:20} mean = {X_raw[c].mean():10.2f}   std = {X_raw[c].std():10.2f}")
print("\nThe data is standardised (mean = 0, std = 1) before clustering.")


# ==================================================================================
# 3. THE K-MEANS ALGORITHM (implemented by hand)
# ==================================================================================
section("3. THE K-MEANS ALGORITHM (IMPLEMENTED BY HAND)")

print("  1. choose k and place the k centroids (k-means++ or at random)")
print("  2. assign every point to the closest centroid")
print("  3. move every centroid to the mean of its assigned points")
print("  4. repeat 2-3 until the centroids stop moving")
print("  WCSS (inertia) = sum of the squared distances of the points to their centroid")


def kmeans_manual(data, k, max_iter=100, tol=1e-4, seed=42):
    """A simple implementation of the Lloyd's (K-Means) algorithm."""
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(len(data), k, replace=False)]

    for iteration in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        new_centroids = centroids.copy()
        for cluster in range(k):
            members = data[labels == cluster]
            if len(members) > 0:
                new_centroids[cluster] = members.mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            print(f"  converged after {iteration + 1} iterations")
            break

    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    inertia = float(distances[np.arange(len(data)), labels].sum())
    return labels, centroids, inertia


manual_labels, manual_centroids, manual_inertia = kmeans_manual(X, k=4, seed=42)
kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=42)
sk_labels = kmeans.fit_predict(X)
sk_inertia = kmeans.inertia_
print(f"\nsklearn K-Means -> inertia (WCSS) = {sk_inertia:.4f}")
print("Manual centroids (standardised space):")
print(pd.DataFrame(manual_centroids,
                   columns=[f"c{c}" for c in range(manual_centroids.shape[1])]).round(3).to_string())


def label_agreement(a, b, k):
    """Best agreement between two labellings, ignoring the numbering of the clusters."""
    from itertools import permutations

    best = 0.0
    for perm in permutations(range(k)):
        mapping = dict(zip(range(k), perm))
        score = np.mean([mapping[x] == y for x, y in zip(a, b)])
        best = max(best, score)
    return best


agreement = label_agreement(manual_labels, sk_labels, 4)
print(f"\nAgreement between the manual and the sklearn implementation "
      f"(ignoring cluster numbering): {agreement * 100:.2f} % of the points")
print(f"Difference in inertia: {abs(manual_inertia - sk_inertia):.6f}")


# ==================================================================================
# 4. CHOOSING THE NUMBER OF CLUSTERS
# ==================================================================================
section("4. CHOOSING THE NUMBER OF CLUSTERS (ELBOW METHOD)")

inertias = []
silhouettes = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(X)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(X, labels_k))
    else:
        silhouettes.append(np.nan)
    print(f"  k = {k:<2} -> inertia = {km.inertia_:10.2f}   "
          f"silhouette = {silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(k_values), inertias, marker="o", color="#4C72B0")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow method")
axes[0].axvline(4, color="crimson", linestyle="--", label="suggested k = 4")
axes[0].legend()

axes[1].plot(list(k_values)[1:], silhouettes[1:], marker="s", color="#C44E52")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette analysis")
axes[1].axvline(4, color="crimson", linestyle="--", label="suggested k = 4")
axes[1].legend()
plt.tight_layout()
save(fig, "lab9_elbow_silhouette.png")

K = 4
final_kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
clusters = final_kmeans.fit_predict(X)
df["Cluster"] = clusters

print(f"\nChosen k = {K}")
print(f"  Silhouette score       : {silhouette_score(X, clusters):.4f}")
print(f"  Calinski-Harabasz index : {calinski_harabasz_score(X, clusters):.2f}")
print(f"  Davies-Bouldin index    : {davies_bouldin_score(X, clusters):.4f}")
print(f"  Inertia                : {final_kmeans.inertia_:.2f}")


# ==================================================================================
# 5. CLUSTER VISUALISATION
# ==================================================================================
section("5. CLUSTER VISUALISATION")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
print(f"PCA explains {pca.explained_variance_ratio_[0] * 100:.1f}% + "
      f"{pca.explained_variance_ratio_[1] * 100:.1f}% = "
      f"{pca.explained_variance_ratio_.sum() * 100:.1f}% of the total variance")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for cluster_id in range(K):
    mask = df["Cluster"] == cluster_id
    axes[0].scatter(df.loc[mask, "PCA1"], df.loc[mask, "PCA2"],
                    s=60, label=f"Cluster {cluster_id}", alpha=0.85)
axes[0].set_title("Clusters in the PCA space")
axes[0].set_xlabel("PCA1")
axes[0].set_ylabel("PCA2")
axes[0].legend()

for cluster_id in range(K):
    mask = df["Cluster"] == cluster_id
    axes[1].scatter(df.loc[mask, "Annual_Income"], df.loc[mask, "Spending_Score"],
                    s=60, label=f"Cluster {cluster_id}", alpha=0.85)
axes[1].set_title("Annual Income vs Spending Score")
axes[1].set_xlabel("Annual Income")
axes[1].set_ylabel("Spending Score")
axes[1].legend()

for cluster_id in range(K):
    mask = df["Cluster"] == cluster_id
    axes[2].scatter(df.loc[mask, "Age"], df.loc[mask, "Purchase_Frequency"],
                    s=60, label=f"Cluster {cluster_id}", alpha=0.85)
axes[2].set_title("Age vs Purchase Frequency")
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Purchase Frequency")
axes[2].legend()
plt.tight_layout()
save(fig, "lab9_cluster_visualisation.png")

centers = scaler.inverse_transform(final_kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=numeric_cols)
centers_df.insert(0, "Cluster", range(K))
print("\nCluster centroids in the original units:")
print(centers_df.round(2).to_string(index=False))


# ==================================================================================
# 6. INTERPRETATION OF THE CLUSTERS
# ==================================================================================
section("6. INTERPRETATION OF THE CLUSTERS")

profile = df.groupby("Cluster")[numeric_cols].mean().round(2)
profile.insert(0, "Customers", df.groupby("Cluster").size().values)
profile["Avg_Purchase_Value"] = (df.groupby("Cluster")["Avg_Basket_Size"].mean()
                                 * df.groupby("Cluster")["Purchase_Frequency"].mean()).round(2)
print("Cluster profile (average of every feature):")
print(profile.to_string())

comparison = pd.crosstab(df["Cluster"], df["Segment"])
print("\nComparison with the original customer segment:")
print(comparison.to_string())

correct = comparison.max(axis=1).sum()
print(f"\nCustomers assigned to the correct segment : {correct} / {len(df)} "
      f"({correct / len(df) * 100:.1f} %)")

fig, ax = plt.subplots(figsize=(9, 6))
angles = np.linspace(0, 2 * np.pi, len(numeric_cols), endpoint=False).tolist()
angles += angles[:1]
scaled_profile = (profile[numeric_cols] - profile[numeric_cols].min()) / \
                 (profile[numeric_cols].max() - profile[numeric_cols].min() + 1e-9)
for cluster_id in range(K):
    values = scaled_profile.loc[cluster_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster_id}")
    ax.fill(angles, values, alpha=0.12)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(numeric_cols, rotation=30, ha="right")
ax.set_title("Cluster profile (normalised)")
ax.legend()
plt.tight_layout()
save(fig, "lab9_cluster_profile.png")

print("\nInterpretation of every cluster (the segment with the most customers wins):")
for cluster_id in range(K):
    row = comparison.loc[cluster_id]
    best_segment = row.idxmax()
    accuracy = row.max() / row.sum() * 100
    print(f"  Cluster {cluster_id}: {best_segment:28} "
          f"({row.max()} customers, {accuracy:.1f}% of the cluster)")

df.to_csv(OUT_DIR / "lab9_clustered_customers.csv", index=False)
print(f"Clustered dataset saved -> {OUT_DIR / 'lab9_clustered_customers.csv'}")

print("\n" + "=" * 78)
print("END OF LAB 9")
print("=" * 78)
