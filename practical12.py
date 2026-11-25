import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris_data = load_iris()
features_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
actual_classes = pd.Series(iris_data.target, name="Actual_Species")

print("Iris dataset loaded successfully!")
print("Dataset shape:", features_df.shape)
print("\nFirst five records:")
print(features_df.head())
print("\nActual classes:", iris_data.target_names)

feature_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(features_df)

pca_model = PCA(n_components=2, random_state=42)
pca_coords = pca_model.fit_transform(scaled_features)

explained_var = pca_model.explained_variance_ratio_
retained_pct = sum(explained_var) * 100

print("\nExplained Variance Ratio (PC1, PC2):", explained_var)
print(f"Total Variance Preserved in 2D: {retained_pct:.2f}%")

pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
pca_df["Actual_Species"] = actual_classes

kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_model.fit_predict(pca_coords)
pca_df["KMeans_Cluster"] = cluster_labels

sil_score = silhouette_score(pca_coords, cluster_labels)
print("\nSilhouette Score (2D K-Means):", round(sil_score, 4))

print("\nComparison of K-Means Clusters vs Actual Species:")
print(pd.crosstab(pca_df["KMeans_Cluster"], pca_df["Actual_Species"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="KMeans_Cluster",
    palette="viridis",
    data=pca_df,
    s=70,
    ax=axes[0],
)
centroids = kmeans_model.cluster_centers_
axes[0].scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    marker="X",
    s=180,
    label="Centroids",
)
axes[0].set_title("K-Means Clusters on 2D PCA (Iris)")
axes[0].legend(loc="best")

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Actual_Species",
    palette="Set1",
    data=pca_df,
    s=70,
    ax=axes[1],
)
axes[1].set_title("Actual Iris Species on 2D PCA")

plt.tight_layout()
plt.show()