import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

wine_dataset = load_wine()
features_df = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)
target_labels = pd.Series(wine_dataset.target, name="Actual_Class")

print("Wine dataset loaded successfully!")
print("Dataset shape:", features_df.shape)
print("\nFirst five records:")
print(features_df.head())

feature_scaler = StandardScaler()
scaled_data = feature_scaler.fit_transform(features_df)

k_clusters = 3
kmeans_clf = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
cluster_preds = kmeans_clf.fit_predict(scaled_data)

sil_avg = silhouette_score(scaled_data, cluster_preds)
print(f"\nSilhouette Score (K={k_clusters}):", round(sil_avg, 4))

pca_reducer = PCA(n_components=2, random_state=42)
pca_points = pca_reducer.fit_transform(scaled_data)
pca_centroids = pca_reducer.transform(kmeans_clf.cluster_centers_)

viz_df = pd.DataFrame(pca_points, columns=["PC1", "PC2"])
viz_df["Cluster"] = cluster_preds
viz_df["Actual_Class"] = target_labels

print("\nCluster vs Actual Class Distribution:")
print(pd.crosstab(viz_df["Cluster"], viz_df["Actual_Class"]))

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="viridis",
    data=viz_df,
    s=75,
)
plt.scatter(
    pca_centroids[:, 0],
    pca_centroids[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids",
)
plt.title(f"K-Means Clustering (K={k_clusters}) Visualized via 2D PCA")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()