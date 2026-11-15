"""
Lab 10: Clustering - Hierarchical Clustering and DBSCAN
Topics:
  Agglomerative clustering
  Dendrogram analysis
  DBSCAN algorithm (implemented by hand and with scikit-learn)
  Density based clustering
  Comparison with K-Means
Run:  python <this file>.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # remove this line if you want to display the plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

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
print("LAB 10 : CLUSTERING - HIERARCHICAL CLUSTERING AND DBSCAN")
print("=" * 78)


# ==================================================================================
# 1. LOADING THE DATA
# ==================================================================================
section("1. LOADING THE DATA")

df = pd.read_csv(DATA_FILE)
numeric_cols = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency",
                "Online_Orders", "Avg_Basket_Size", "Recency_Days"]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df[numeric_cols])
print(f"Dataset shape : {df.shape}")
print(f"Features      : {numeric_cols}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


def plot_clusters(embedding, labels, title, filename, centroids=None):
    """Scatter plot of a 2-D embedding coloured by cluster."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels,
                         cmap="tab10", s=60, edgecolor="k", alpha=0.85)
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=220,
                   c="red", edgecolor="black", label="centroids")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
    plt.tight_layout()
    save(fig, filename)


# ==================================================================================
# 2. AGGLOMERATIVE (HIERARCHICAL) CLUSTERING
# ==================================================================================
section("2. AGGLOMERATIVE CLUSTERING")

print("Every point starts in its own cluster, the two closest clusters are merged")
print("repeatedly until one cluster is left.")
print("  ward     : merge that minimises the increase of the variance (WCSS)")
print("  complete : distance between the farthest points of the two clusters")
print("  average  : average distance between all the points of the two clusters")
print("  single   : distance between the closest points of the two clusters")


def make_agglomerative(linkage_type, n_clusters, metric="euclidean"):
    """Build an AgglomerativeClustering object (compatible with old/new sklearn)."""
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type,
                                       metric=metric)
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type,
                                       affinity=metric)


K = 4
linkage_results = []
for linkage_type in ["ward", "complete", "average", "single"]:
    model = make_agglomerative(linkage_type, K)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    linkage_results.append({"linkage": linkage_type, "silhouette": score,
                            "clusters": len(set(labels))})
    print(f"  linkage = {linkage_type:9} -> silhouette = {score:.4f}")

linkage_df = pd.DataFrame(linkage_results)
best_linkage = linkage_df.loc[linkage_df["silhouette"].idxmax(), "linkage"]
print(f"\nBest linkage method : {best_linkage}")

agg_model = make_agglomerative(best_linkage, K)
agg_labels = agg_model.fit_predict(X)
df["Agglomerative_Cluster"] = agg_labels
print(f"Cluster sizes : {np.bincount(agg_labels).tolist()}")


# ==================================================================================
# 3. DENDROGRAM ANALYSIS
# ==================================================================================
section("3. DENDROGRAM ANALYSIS")

Z = linkage(X, method=best_linkage, metric="euclidean")
print(f"The linkage matrix has the shape {Z.shape} "
      f"(one row per merge: [i, j, distance, number of points])")
print("First 5 merges:")
print(pd.DataFrame(Z[:5], columns=["Cluster 1", "Cluster 2", "Distance", "Size"]).round(3).to_string(index=False))

coph_corr, coph_dists = cophenet(Z, pdist(X))
print(f"\nCophenetic correlation coefficient : {coph_corr:.4f}")
print("  (1.0 = the dendrogram reproduces the original distances perfectly)")

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=40, color_threshold=0)
ax.set_title(f"Dendrogram - agglomerative clustering ({best_linkage} linkage)")
ax.set_xlabel("Customers")
ax.set_ylabel("Distance")
plt.tight_layout()
save(fig, "lab10_dendrogram.png")

for n in [2, 3, 4, 5]:
    cut_labels = fcluster(Z, t=n, criterion="maxclust")
    score = silhouette_score(X, cut_labels)
    print(f"  cut the dendrogram into {n} clusters -> silhouette = {score:.4f}")

cut_labels = fcluster(Z, t=K, criterion="maxclust")
print(f"\nClusters obtained by cutting the dendrogram into {K} groups: {np.bincount(cut_labels)[1:].tolist()}")

plot_clusters(X_pca, agg_labels, f"Agglomerative clustering ({best_linkage})",
              "lab10_agglomerative_clusters.png")


# ==================================================================================
# 4. DBSCAN - DENSITY BASED CLUSTERING
# ==================================================================================
section("4. DBSCAN (DENSITY BASED CLUSTERING)")

print("  eps         : neighbourhood radius. A point with at least min_samples")
print("                neighbours inside this radius is a CORE point.")
print("  min_samples : minimum number of points (core point included) of a cluster")
print("  core point   -> at least min_samples neighbours within eps")
print("  border point -> not a core point but neighbour of a core point")
print("  noise point  -> not reachable from any core point (label -1)")
print("The number of clusters is not fixed and the clusters can have any shape.")


def region_query(distance_matrix, index, eps):
    """All the points within the radius eps of the point `index`."""
    return list(np.where(distance_matrix[index] <= eps)[0])


def dbscan_manual(data, eps, min_samples):
    """A simple implementation of the DBSCAN algorithm."""
    distance_matrix = squareform(pdist(data))
    n = len(data)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for point in range(n):
        if visited[point]:
            continue
        visited[point] = True
        neighbours = region_query(distance_matrix, point, eps)

        if len(neighbours) < min_samples:
            labels[point] = -1
            continue

        labels[point] = cluster_id
        queue = list(neighbours)
        while queue:
            other = queue.pop()
            if not visited[other]:
                visited[other] = True
                new_neighbours = region_query(distance_matrix, other, eps)
                if len(new_neighbours) >= min_samples:
                    queue.extend(new_neighbours)
            if labels[other] == -1:
                labels[other] = cluster_id
        cluster_id += 1

    return labels


rng = np.random.default_rng(42)
dense = rng.normal(loc=[0, 0], scale=0.35, size=(120, 2))
sparse = rng.normal(loc=[3.5, 3.5], scale=0.95, size=(70, 2))
noise = rng.uniform(low=-4, high=7, size=(45, 2))
synthetic = np.vstack([dense, sparse, noise])
print(f"Synthetic data set : dense group (120), sparse group (70), noise points (45)")

manual_dbscan = dbscan_manual(synthetic, eps=0.4, min_samples=6)
sk_dbscan = DBSCAN(eps=0.4, min_samples=6).fit_predict(synthetic)
print(f"\nManual DBSCAN  : clusters = {len(set(manual_dbscan)) - (1 if -1 in manual_dbscan else 0)}, "
      f"noise points = {int(np.sum(manual_dbscan == -1))}")
print(f"sklearn DBSCAN : clusters = {len(set(sk_dbscan)) - (1 if -1 in sk_dbscan else 0)}, "
      f"noise points = {int(np.sum(sk_dbscan == -1))}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(synthetic[:, 0], synthetic[:, 1], c=manual_dbscan, cmap="tab10",
                 s=30, edgecolor="k", linewidths=0.3)
axes[0].set_title("DBSCAN (manual implementation)")
axes[1].scatter(synthetic[:, 0], synthetic[:, 1], c=sk_dbscan, cmap="tab10",
                 s=30, edgecolor="k", linewidths=0.3)
axes[1].set_title("DBSCAN (scikit-learn)")
plt.tight_layout()
save(fig, "lab10_dbscan_synthetic.png")

print("\nDBSCAN on the customer segments data:")
dbscan_results = []
for eps in [0.4, 0.6, 0.8, 1.0, 1.2]:
    for min_samples in [3, 5, 10]:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        score = silhouette_score(X[labels != -1], labels[labels != -1]) if n_clusters > 1 else -1
        dbscan_results.append({"eps": eps, "min_samples": min_samples,
                               "clusters": n_clusters, "noise": n_noise,
                               "silhouette": score})

dbscan_df = pd.DataFrame(dbscan_results)
print(dbscan_df.round(4).to_string(index=False))

best_row = dbscan_df.loc[dbscan_df["silhouette"].idxmax()]
best_eps, best_min = best_row["eps"], best_row["min_samples"]
print(f"\nBest DBSCAN setting : eps = {best_eps}, min_samples = {best_min} "
      f"(silhouette = {best_row['silhouette']:.4f})")

dbscan_labels = DBSCAN(eps=best_eps, min_samples=best_min).fit_predict(X)
df["DBSCAN_Cluster"] = dbscan_labels
print(f"Clusters found : {best_row['clusters']}, noise points : {int(np.sum(dbscan_labels == -1))}")
plot_clusters(X_pca, dbscan_labels,
              f"DBSCAN (eps={best_eps}, min_samples={best_min})",
              "lab10_dbscan_customers.png")


# ==================================================================================
# 5. COMPARISON WITH K-MEANS
# ==================================================================================
section("5. COMPARISON WITH K-MEANS")

kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

comparison = pd.DataFrame([
    {
        "Algorithm": "K-Means",
        "Clusters found": len(set(kmeans_labels)),
        "Noise points": 0,
        "Silhouette": silhouette_score(X, kmeans_labels),
        "ARI vs K-Means": 1.0,
        "Needs k": "yes",
        "Cluster shape": "convex / spherical",
    },
    {
        "Algorithm": f"Agglomerative ({best_linkage})",
        "Clusters found": len(set(agg_labels)),
        "Noise points": 0,
        "Silhouette": silhouette_score(X, agg_labels),
        "ARI vs K-Means": adjusted_rand_score(kmeans_labels, agg_labels),
        "Needs k": "yes (or a distance cut)",
        "Cluster shape": "any",
    },
    {
        "Algorithm": f"DBSCAN (eps={best_eps}, min={best_min})",
        "Clusters found": best_row["clusters"],
        "Noise points": int(np.sum(dbscan_labels == -1)),
        "Silhouette": best_row["silhouette"],
        "ARI vs K-Means": adjusted_rand_score(kmeans_labels, dbscan_labels),
        "Needs k": "no",
        "Cluster shape": "any",
    },
])
print(comparison.round(4).to_string(index=False))


fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (labels, title) in zip(axes, [
        (kmeans_labels, "K-Means"),
        (agg_labels, f"Agglomerative ({best_linkage})"),
        (dbscan_labels, f"DBSCAN (eps={best_eps}, min={best_min})")]):
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10",
                         s=55, edgecolor="k", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
plt.tight_layout()
save(fig, "lab10_comparison.png")

df.to_csv(OUT_DIR / "lab10_clustered_customers.csv", index=False)
print(f"Clustered dataset saved -> {OUT_DIR / 'lab10_clustered_customers.csv'}")

print("\n" + "=" * 78)
print("END OF LAB 10")
print("=" * 78)
