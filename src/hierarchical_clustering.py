# Step #21: Perform Hierarchical Clustering Analysis

linkage_matrix = linkage(
    X_pca,
    method='complete',
    metric='euclidean'
)

print("Hierarchical clustering complete.")

# Step #22: Construct and Interpret the Dendrogram

plt.figure(figsize=(16,8))

dendrogram(
    linkage_matrix,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Distance")

plt.savefig("Wine_Clustering_PCA_KMeans_Hierarchical/reports/figures/hierarchical_clustering_dendrogram.png", dpi=300, bbox_inches="tight")

plt.show()

# Step #23: Generate Hierarchical Cluster Assignments

hier_clusters = fcluster(
    linkage_matrix,
    t=3,
    criterion='maxclust'
)

df["Hierarchical_Cluster"] = hier_clusters

print(df["Hierarchical_Cluster"].value_counts())

# Step #24: Assess Dendrogram Reliability with Cophenetic Correlation

coph_corr, coph_dist = cophenet(
    linkage_matrix,
    pdist(X_pca)
)

print("Cophenetic Correlation:", round(coph_corr,4))

# Step #25: Compare K-Means and Hierarchical Clustering Results

comparison = pd.crosstab(
    df["Cluster"],
    df["Hierarchical_Cluster"]
)


comparison
