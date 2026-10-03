# Step #13: Determine Optimal K Using the Elbow Method

inertia = []

k_values = range(2,11)

for k in k_values:
    
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    km.fit(X_pca)
    
    inertia.append(km.inertia_)

plt.figure(figsize=(10,6))

plt.plot(
    k_values,
    inertia,
    marker='o'
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.savefig("Wine_Clustering_PCA_KMeans_Hierarchical/reports/figures/elbow_method.png", dpi=300, bbox_inches="tight")

plt.show()

# Step #14: Evaluate Cluster Quality with Silhouette Analysis

scores = []

for k in range(2,11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels)

    scores.append(score)

silhouette_df = pd.DataFrame({
    "K": range(2,11),
    "Silhouette Score": scores
})

silhouette_df

# Step #15: Visualize Silhouette Performance Across K Values

plt.figure(figsize=(10,6))

sns.lineplot(
    x=range(2,11),
    y=scores,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")

plt.savefig("Wine_Clustering_PCA_KMeans_Hierarchical/reports/figures/silhouette_analysis.png", dpi=300, bbox_inches="tight")

plt.show()

# Step #16: Train the Optimal K-Means Clustering Model

best_k = silhouette_df.loc[
    silhouette_df["Silhouette Score"].idxmax(),
    "K"
]

print("Best K:", best_k)

kmeans = KMeans(
    n_clusters=int(best_k),
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)

df["Cluster"] = clusters

# Step #17: Visualize K-Means Cluster Separation

plt.figure(figsize=(10,7))

sns.scatterplot(
    x=X_pca_2d[:,0],
    y=X_pca_2d[:,1],
    hue=df["Cluster"],
    palette="Set1",
    s=100
)

plt.title("K-Means Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("Wine_Clustering_PCA_KMeans_Hierarchical/reports/figures/K-Means_clusters.png", dpi=300, bbox_inches="tight")

plt.show()

# Step #18: Analyze Cluster Membership Distribution

cluster_counts = df["Cluster"].value_counts()

print(cluster_counts)

# Step #19: Develop Cluster Profile Summaries

cluster_profiles = df.groupby("Cluster").mean()

cluster_profiles

# Step #20: Visualize Cluster Characteristics Using Heatmaps

plt.figure(figsize=(15,8))

sns.heatmap(
    cluster_profiles,
    annot=True,
    cmap="YlGnBu"
)

plt.title("Cluster Feature Profiles")

plt.savefig("Wine_Clustering_PCA_KMeans_Hierarchical/reports/figures/cluster_feature_profiles.png", dpi=300, bbox_inches="tight")

plt.show()

