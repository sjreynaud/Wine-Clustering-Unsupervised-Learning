# Step #7: Conduct Initial Principal Component Analysis (PCA)

pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_

print(explained_variance)

# Step #8: Generate PCA Scree Plot

plt.figure(figsize=(10,6))

plt.plot(
    range(1, len(explained_variance)+1),
    np.cumsum(explained_variance),
    marker='o'
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("PCA Cumulative Explained Variance")
plt.axhline(y=0.80, color='red', linestyle='--')

plt.savefig("Wine_Clustering_PCA_KMeans_Hierarchical/reports/figures/pca_cumulative_explained_variance.png", dpi=300, bbox_inches="tight")

plt.show()

# Step #9: Select Components Retaining 80% Variance

cumulative_variance = np.cumsum(explained_variance)

n_components = np.argmax(cumulative_variance >= 0.80) + 1

print("Number of Components for 80% Variance:", n_components)

# Step #10: Apply PCA Dimensionality Reduction

pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_scaled)

print("Original Shape:", X_scaled.shape)
print("Reduced Shape:", X_pca.shape)

# Step #11: Interpret Principal Component Loadings

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(n_components)],
    index=df.columns
)

loadings

# Step #12: Visualize PCA Component Structure

pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10,7))

plt.scatter(
    X_pca_2d[:,0],
    X_pca_2d[:,1],
    alpha=0.7
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Dataset PCA Projection")

plt.savefig("Wine_Clustering_PCA_KMeans_Hierarchical/reports/figures/wine_dataset-PCA_projection.png", dpi=300, bbox_inches="tight")

plt.show()
