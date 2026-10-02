# Wine Clustering Using PCA, K-Means, and Hierarchical Clustering
 
## Project Overview
 
This project applies unsupervised machine learning techniques to a wine dataset to discover hidden patterns and natural groupings among wine observations. The analysis combines Principal Component Analysis (PCA) for dimensionality reduction, K-Means Clustering for cluster formation, and Hierarchical Clustering for cluster validation and comparison.
 
The objective is to reduce feature complexity while preserving most of the dataset's variability, identify meaningful clusters, and evaluate clustering performance using multiple validation techniques.
 
---
 
## Objectives
 
- Perform exploratory data analysis (EDA)
- Investigate assumptions for Principal Component Analysis (PCA)
- Standardize numerical features
- Reduce dimensionality using PCA
- Retain at least 80% of the total dataset variance
- Identify the optimal number of clusters using the Elbow Method
- Evaluate cluster quality using Silhouette Analysis
- Build and interpret a K-Means clustering model
- Perform Hierarchical Clustering using Complete Linkage
- Visualize clustering structures with PCA plots and dendrograms
- Compare K-Means and Hierarchical Clustering solutions
- Interpret cluster characteristics and model findings
 
---
 
## Dataset
 
**Dataset:** Wine Clustering Dataset
 
**Source:** Kaggle
 
The dataset contains physicochemical properties of wines, including:
 
- Alcohol
- Malic Acid
- Ash
- Ash Alcanity
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280
- Proline
 
---
 
## Project Workflow
 
### 1. Data Preparation
- Import required libraries
- Load dataset
- Explore data structure
- Check for missing values and duplicates
- Generate descriptive statistics
 
### 2. Principal Component Analysis (PCA)
- Analyze feature correlations
- Investigate PCA assumptions
- Standardize features
- Perform PCA
- Determine components explaining at least 80% variance
- Interpret principal component loadings
- Visualize observations in PCA space
 
### 3. K-Means Clustering
- Apply Elbow Method
- Calculate Silhouette Scores
- Select optimal number of clusters
- Train final K-Means model
- Visualize cluster separation
- Analyze cluster membership distribution
- Develop cluster profiles
 
### 4. Hierarchical Clustering
- Perform Complete-Linkage Hierarchical Clustering
- Construct dendrogram
- Generate hierarchical cluster assignments
- Evaluate model quality using Cophenetic Correlation
 
### 5. Model Comparison
- Compare K-Means and Hierarchical Clustering results
- Assess cluster agreement
- Summarize key findings
 
---
 
## Technologies Used
 
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
 
---
 
## Repository Structure
 
```text
wine-clustering-unsupervised-learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ └── wine-clustering.csv
│
├── notebooks/
│ └── Wine_Clustering_PCA_KMeans_Hierarchical.ipynb
│
├── reports/
│ ├── Assignment_7_Report.pdf
│ │
│ └── figures/
│ ├── correlation_matrix.png
│ ├── feature_distributions.png
│ ├── pca_cumulative_variance.png
│ ├── pca_scatter_plot.png
│ ├── elbow_method.png
│ ├── silhouette_analysis.png
│ ├── kmeans_clusters.png
│ ├── cluster_profiles_heatmap.png
│ └── hierarchical_dendrogram.png
│
├── results/
│ ├── pca_loadings.csv
│ ├── silhouette_scores.csv
│ ├── cluster_profiles.csv
│ ├── cluster_comparison.csv
│ └── final_summary.txt
│
└── src/
├── data_preprocessing.py
├── pca_analysis.py
├── kmeans_clustering.py
├── hierarchical_clustering.py
└── visualization.py
```
 
---
