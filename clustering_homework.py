import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
import time
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 61
np.random.seed(RANDOM_SEED)


# SECTION 4: DATA PREPROCESSING

def load_and_preprocess_data():
    print("-" * 80)
    print("SECTION 4: DATA PREPROCESSING")
    
    # Load dataset
    wine = load_wine()
    X = wine.data
    feature_names = wine.feature_names
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=feature_names)
    
    # 1. Report dataset shape
    print(f"\n1. Dataset Shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features: {X.shape[1]}")
    
    # 2. Compute and report mean and standard deviation of each feature
    print("\n2. Feature Statistics (Mean and Standard Deviation):")
    stats_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean': np.mean(X, axis=0),
        'Std Dev': np.std(X, axis=0)
    })
    print(stats_df.to_string(index=False))
    
    # 3. Verify absence of missing values
    missing_values = np.sum(np.isnan(X))
    print(f"\n3. Missing Values Check: {missing_values} missing values found")
    if missing_values == 0:
        print("   Dataset is complete with no missing values.")
    
    # 4. Create three versions of the dataset
    print("\n4. Creating Three Versions of the Dataset:")
    
    # Raw (unscaled) data
    X_raw = X.copy()
    print("   - Raw (unscaled) data: Created")
    
    # Standardized data using Z-score normalization
    scaler_standard = StandardScaler()
    X_standardized = scaler_standard.fit_transform(X)
    print("   - Standardized data (Z-score): Created")
    
    # Min-Max normalized data
    scaler_minmax = MinMaxScaler()
    X_minmax = scaler_minmax.fit_transform(X)
    print("   - Min-Max normalized data: Created")
    
    return {
        'raw': X_raw,
        'standardized': X_standardized,
        'minmax': X_minmax,
        'feature_names': feature_names,
        'dataframe': df
    }


# SECTION 5: EXPLORATORY DATA ANALYSIS (EDA) AND VISUALIZATION

def perform_eda(data_dict, save_plots=True):

    print("\n" + "-" * 80)
    print("SECTION 5: EXPLORATORY DATA ANALYSIS (EDA)")
    
    df = data_dict['dataframe']
    feature_names = data_dict['feature_names']
    X_standardized = data_dict['standardized']
    
    # 5.1 General Dataset Exploration
    print("\n5.1 General Dataset Exploration")
    print("-" * 40)
    
    # Summary statistics table
    print("\nSummary Statistics Table:")
    summary_stats = df.describe().T
    summary_stats['variance'] = df.var()
    print(summary_stats.to_string())
    
    if save_plots:
        # Plot histograms for five selected features
        selected_features = list(feature_names[:5])
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle('Histograms of Selected Features', fontsize=14)
        
        for i, feature in enumerate(selected_features):
            axes[i].hist(df[feature], bins=20, edgecolor='black', alpha=0.7)
            axes[i].set_title(feature)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('eda_histograms.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n   Histograms saved to 'eda_histograms.png'")
        
        # Correlation matrix heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5)
        plt.title('Correlation Matrix Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig('eda_correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   Correlation matrix saved to 'eda_correlation_matrix.png'")
        
        # Scatter plots for two meaningful pairs of features (based on correlation analysis)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Scatter Plots of Highly Correlated Feature Pairs', fontsize=14)
        
        # Pair 1: total_phenols vs flavanoids (highest positive correlation: r = 0.86)
        axes[0].scatter(df['total_phenols'], df['flavanoids'], alpha=0.6, c='steelblue')
        axes[0].set_xlabel('Total Phenols')
        axes[0].set_ylabel('Flavanoids')
        axes[0].set_title('Total Phenols vs Flavanoids (r = 0.86)')
        
        # Pair 2: malic_acid vs hue (strongest negative correlation: r = -0.56)
        axes[1].scatter(df['malic_acid'], df['hue'], alpha=0.6, c='darkorange')
        axes[1].set_xlabel('Malic Acid')
        axes[1].set_ylabel('Hue')
        axes[1].set_title('Malic Acid vs Hue (r = -0.56)')
        
        plt.tight_layout()
        plt.savefig('eda_scatter_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   Scatter plots saved to 'eda_scatter_plots.png'")
    
    # 5.2 Dimensionality Reduction for Visualization
    print("\n5.2 Dimensionality Reduction (PCA) for Visualization")
    print("-" * 40)
    
    pca_viz = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca_viz.fit_transform(X_standardized)
    
    print(f"   Explained variance ratio: {pca_viz.explained_variance_ratio_}")
    print(f"   Total variance explained: {sum(pca_viz.explained_variance_ratio_):.4f}")
    
    if save_plots:
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.xlabel(f'Principal Component 1 ({pca_viz.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca_viz.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization of Wine Dataset (2D)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_pca_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   PCA scatter plot saved to 'eda_pca_visualization.png'")
    
    return X_pca


# PART 1: DISTANCE BEHAVIOR IN HIGH-DIMENSIONAL SPACE

def analyze_distance_behavior(data_dict):

    print("\n" + "-" * 80)
    print("PART 1: DISTANCE BEHAVIOR IN HIGH-DIMENSIONAL SPACE")
    
    X = data_dict['standardized']
    
    print("\nComputing pairwise distances using different metrics...")
    
    # Compute pairwise distances
    dist_euclidean = pdist(X, metric='euclidean')
    dist_manhattan = pdist(X, metric='cityblock')
    dist_cosine = pdist(X, metric='cosine')
    
    # Compute statistics for each metric
    results = []
    
    for metric_name, distances in [
        ('Euclidean', dist_euclidean),
        ('Manhattan', dist_manhattan),
        ('Cosine', dist_cosine)
    ]:
        stats = {
            'Metric': metric_name,
            'Mean': np.mean(distances),
            'Std Dev': np.std(distances),
            'Min': np.min(distances),
            'Max': np.max(distances),
            'Coefficient of Variation': np.std(distances) / np.mean(distances)
        }
        results.append(stats)
    
    results_df = pd.DataFrame(results)
    print("\nDistance Statistics by Metric:")
    print(results_df.to_string(index=False))
    
    # Analysis of distance concentration
    print("\nDistance Concentration Analysis:")
    for _, row in results_df.iterrows():
        cv = row['Coefficient of Variation']
        print(f"   {row['Metric']}: CV = {cv:.4f} ", end="")
        if cv < 0.2:
            print("(High concentration - distances are similar)")
        elif cv < 0.5:
            print("(Moderate variation)")
        else:
            print("(Good separation)")
    
    return results_df


# PART 2: IMPACT OF FEATURE SCALING ON K-MEANS

def analyze_scaling_impact(data_dict, k=3,alpha=0.7):
    print("\n" + "-" * 80)
    print("PART 2: IMPACT OF FEATURE SCALING ON K-MEANS")
    
    results = []
    
    for data_type, X in [
        ('Raw', data_dict['raw']),
        ('Standardized', data_dict['standardized']),
        ('Min-Max', data_dict['minmax'])
    ]:
        # Apply K-Means
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Compute metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)
        
        results.append({
            'Data Type': data_type,
            'Inertia': inertia,
            'Silhouette Score': silhouette,
            'Iterations': kmeans.n_iter_
        })
    
    results_df = pd.DataFrame(results)
    
    # -----------------------------
    # Normalize metrics
    # -----------------------------
    
    # Silhouette (higher better)
    sil_norm = (results_df['Silhouette Score'] - results_df['Silhouette Score'].min()) / \
               (results_df['Silhouette Score'].max() - results_df['Silhouette Score'].min())
    
    # Inertia (lower better → invert)
    inertia_norm = (results_df['Inertia'].max() - results_df['Inertia']) / \
                   (results_df['Inertia'].max() - results_df['Inertia'].min())
    
    # Combined score
    results_df['Combined Score'] = alpha * sil_norm + (1 - alpha) * inertia_norm
    
    print(f"\nK-Means Clustering Results (k={k}):")
    print(results_df.to_string(index=False))
    
    best_idx = results_df['Combined Score'].idxmax()
    best_method = results_df.loc[best_idx, 'Data Type']
    
    print(f"\nBest scaling method (combined score): {best_method}")
    
    return results_df
    
# PART 3: K-MEANS IMPLEMENTATION FROM SCRATCH

class KMeansFromScratch:
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def _initialize_centroids(self, X):
        """Initialize centroids using random selection from data points."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids by computing mean of assigned points."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty cluster by reinitializing
                new_centroids[k] = X[np.random.randint(X.shape[0])]
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Compute sum of squared distances to nearest centroid."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                distances = np.sum((cluster_points - self.centroids[k]) ** 2)
                inertia += distances
        return inertia
    
    def fit(self, X):
        """Fit K-Means to the data."""
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assignment step
            labels = self._assign_clusters(X)
            
            # Update step
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            self.n_iter_ = iteration + 1
            
            if centroid_shift < self.tol:
                break
        
        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for data."""
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def compare_kmeans_implementations(data_dict, k=3):
    """
    Compare custom K-Means implementation with scikit-learn.
    """
    print("\n" + "-" * 80)
    print("PART 3: K-MEANS IMPLEMENTATION FROM SCRATCH")
    
    X = data_dict['standardized']
    
    # Custom implementation
    print("\nRunning custom K-Means implementation...")
    start_time = time.time()
    kmeans_custom = KMeansFromScratch(n_clusters=k, random_state=RANDOM_SEED)
    labels_custom = kmeans_custom.fit_predict(X)
    time_custom = time.time() - start_time
    
    # Scikit-learn implementation
    print("Running scikit-learn K-Means implementation...")
    start_time = time.time()
    kmeans_sklearn = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=1, init='random')
    labels_sklearn = kmeans_sklearn.fit_predict(X)
    time_sklearn = time.time() - start_time
    
    # Compare results
    results = {
        'Metric': ['Inertia', 'Number of Iterations', 'Runtime (seconds)', 'Silhouette Score'],
        'Custom Implementation': [
            kmeans_custom.inertia_,
            kmeans_custom.n_iter_,
            time_custom,
            silhouette_score(X, labels_custom)
        ],
        'Scikit-learn': [
            kmeans_sklearn.inertia_,
            kmeans_sklearn.n_iter_,
            time_sklearn,
            silhouette_score(X, labels_sklearn)
        ]
    }
    
    results_df = pd.DataFrame(results)
    print(f"\nComparison of K-Means Implementations (k={k}):")
    print(results_df.to_string(index=False))
    
    return results_df


# PART 4: HIERARCHICAL CLUSTERING STABILITY ANALYSIS

def analyze_hierarchical_clustering(data_dict):

    print("\n" + "-" * 80)
    print("PART 4: HIERARCHICAL CLUSTERING STABILITY ANALYSIS")
    
    X = data_dict['standardized']
    
    linkage_methods = ['single', 'complete', 'average', 'ward']
    results = []
    
    print("\nComputing cophenetic correlation for each linkage method...")
    
    # Compute pairwise distances for cophenetic correlation
    pdist_matrix = pdist(X)
    
    for method in linkage_methods:
        # Perform hierarchical clustering
        Z = linkage(X, method=method)
        
        # Compute cophenetic correlation coefficient
        cophenetic_dists, _ = cophenet(Z, pdist_matrix)
        coph_corr = cophenetic_dists
        
        # Also compute silhouette score by cutting at k=3 clusters
        agg = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = agg.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        
        results.append({
            'Linkage Method': method,
            'Cophenetic Correlation': coph_corr,
            'Silhouette Score (k=3)': silhouette
        })
    
    results_df = pd.DataFrame(results)
    
    # Rank by cophenetic correlation (stability indicator)
    results_df['Stability Rank'] = results_df['Cophenetic Correlation'].rank(ascending=False).astype(int)
    results_df = results_df.sort_values('Stability Rank')
    
    print("\nHierarchical Clustering Results (ranked by stability):")
    print(results_df.to_string(index=False))
    
    # Programmatically select the most stable linkage method
    best_method = results_df.loc[results_df['Stability Rank'] == 1, 'Linkage Method'].values[0]
    best_coph = results_df.loc[results_df['Stability Rank'] == 1, 'Cophenetic Correlation'].values[0]
    
    print(f"\nMost stable linkage method: {best_method}")
    print(f"Cophenetic correlation coefficient: {best_coph:.4f}")
    
    return results_df, best_method


# PART 5: AUTOMATED DENSITY-BASED CLUSTERING (DBSCAN)

def automated_dbscan_clustering(data_dict):

    print("\n" + "-" * 80)
    print("PART 5: AUTOMATED DENSITY-BASED CLUSTERING")
    
    X = data_dict['standardized']
    k = 5  # k for k-nearest neighbors
    
    # Step 1: Compute k-nearest neighbor distances
    print(f"\nStep 1: Computing {k}-nearest neighbor distances...")
    
    nbrs = NearestNeighbors(n_neighbors=k + 1)  # +1 because point itself is included
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Get k-th nearest neighbor distance (excluding self)
    k_distances = distances[:, k]
    k_distances_sorted = np.sort(k_distances)
    
    # Step 2: Estimate epsilon using statistical heuristic
    print("Step 2: Estimating epsilon parameter...")
    
    # Use the "elbow" method approximation via numerical derivative
    # Find point of maximum curvature
    diffs = np.diff(k_distances_sorted)
    second_diffs = np.diff(diffs)
    
    # Find elbow point (where second derivative is maximum)
    elbow_idx = np.argmax(second_diffs) + 1
    eps_estimated = k_distances_sorted[elbow_idx]
    
    # Alternative: Use statistics-based heuristic (mean + std)
    eps_statistical = np.mean(k_distances) + np.std(k_distances)
    
    # Use the minimum of the two for a more conservative estimate
    eps_final = min(eps_estimated, eps_statistical)
    
    print(f"   Elbow-based epsilon: {eps_estimated:.4f}")
    print(f"   Statistical epsilon: {eps_statistical:.4f}")
    print(f"   Selected epsilon: {eps_final:.4f}")
    
    # Step 3: Apply DBSCAN
    print(f"\nStep 3: Applying DBSCAN with eps={eps_final:.4f}, min_samples={k}...")
    
    dbscan = DBSCAN(eps=eps_final, min_samples=k)
    labels = dbscan.fit_predict(X)
    
    # Step 4: Report results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nStep 4: Results:")
    print(f"   Number of clusters detected: {n_clusters}")
    print(f"   Number of noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    # Step 5: If DBSCAN fails to produce meaningful clusters, modify feature space
    if n_clusters < 2 or n_noise > len(labels) * 0.5:
        print("\nDBSCAN did not produce meaningful clusters. Trying with PCA-reduced space...")
        
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=5, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X)
        
        # Recompute k-distances
        nbrs_pca = NearestNeighbors(n_neighbors=k + 1)
        nbrs_pca.fit(X_pca)
        distances_pca, _ = nbrs_pca.kneighbors(X_pca)
        k_distances_pca = distances_pca[:, k]
        
        # Re-estimate epsilon
        eps_pca = np.mean(k_distances_pca) + 0.5 * np.std(k_distances_pca)
        
        print(f"   New epsilon (PCA space): {eps_pca:.4f}")
        
        dbscan_pca = DBSCAN(eps=eps_pca, min_samples=k)
        labels_pca = dbscan_pca.fit_predict(X_pca)
        
        n_clusters_pca = len(set(labels_pca)) - (1 if -1 in labels_pca else 0)
        n_noise_pca = list(labels_pca).count(-1)
        
        print(f"   Clusters in PCA space: {n_clusters_pca}")
        print(f"   Noise points in PCA space: {n_noise_pca} ({n_noise_pca/len(labels_pca)*100:.1f}%)")
        
        if n_clusters_pca > n_clusters:
            labels = labels_pca
            n_clusters = n_clusters_pca
            n_noise = n_noise_pca
            eps_final = eps_pca
            print("   Using PCA-based clustering results.")
    
    # Compute silhouette score if valid clusters exist
    if n_clusters >= 2 and n_noise < len(labels):
        valid_mask = labels != -1
        if sum(valid_mask) > n_clusters:
            silhouette = silhouette_score(X[valid_mask] if len(X) == len(labels) else X_pca[valid_mask], 
                                          labels[valid_mask])
            print(f"\nSilhouette Score (excluding noise): {silhouette:.4f}")
    
    return {
        'eps': eps_final,
        'min_samples': k,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'labels': labels
    }


# PART 6: FULLY AUTOMATED CLUSTERING SYSTEM

def automated_clustering_system(data_dict):
    """
    Fully automated clustering system that selects the best algorithm and parameters.
    """
    print("\n" + "-" * 80)
    print("PART 6: FULLY AUTOMATED CLUSTERING SYSTEM")
    
    results = []
    
    # Preprocessing options
    preprocessing_options = {
        'raw': data_dict['raw'],
        'standardized': data_dict['standardized'],
        'minmax': data_dict['minmax']
    }
    
    # K values to try
    k_values = [2, 3, 4, 5]
    
    print("\nEvaluating multiple clustering configurations...")
    print("-" * 60)
    
    # 1. K-Means with different preprocessing and k values
    print("\n[1/3] Testing K-Means configurations...")
    for prep_name, X in preprocessing_options.items():
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                labels = kmeans.fit_predict(X)
                silhouette = silhouette_score(X, labels)
                
                results.append({
                    'Algorithm': 'K-Means',
                    'Preprocessing': prep_name,
                    'Parameters': f'k={k}',
                    'Silhouette Score': silhouette,
                    'n_clusters': k,
                    'labels': labels.copy()
                })
            except Exception as e:
                continue
    
    # 2. Agglomerative Clustering with different linkage methods
    print("[2/3] Testing Agglomerative Clustering configurations...")
    linkage_methods = ['single', 'complete', 'average', 'ward']
    for prep_name, X in preprocessing_options.items():
        for k in k_values:
            for linkage_method in linkage_methods:
                try:
                    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
                    labels = agg.fit_predict(X)
                    silhouette = silhouette_score(X, labels)
                    
                    results.append({
                        'Algorithm': 'Agglomerative',
                        'Preprocessing': prep_name,
                        'Parameters': f'k={k}, linkage={linkage_method}',
                        'Silhouette Score': silhouette,
                        'n_clusters': k,
                        'labels': labels.copy()
                    })
                except Exception as e:
                    continue
    
    # 3. DBSCAN with parameter tuning
    print("[3/3] Testing DBSCAN configurations...")
    eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    min_samples_values = [3, 5, 7]
    
    for prep_name, X in preprocessing_options.items():
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # Skip configurations with too few clusters or too much noise
                    if n_clusters < 2 or n_noise > len(labels) * 0.5:
                        continue
                    
                    # Compute silhouette on non-noise points
                    valid_mask = labels != -1
                    if sum(valid_mask) > n_clusters:
                        silhouette = silhouette_score(X[valid_mask], labels[valid_mask])
                        
                        results.append({
                            'Algorithm': 'DBSCAN',
                            'Preprocessing': prep_name,
                            'Parameters': f'eps={eps}, min_samples={min_samples}',
                            'Silhouette Score': silhouette,
                            'n_clusters': n_clusters,
                            'labels': labels.copy()
                        })
                except Exception as e:
                    continue
    
    # Convert to DataFrame and find best configuration
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid clustering configurations found.")
        return None
    
    # Sort by silhouette score
    results_df = results_df.sort_values('Silhouette Score', ascending=False)
    
    # Display top 10 configurations
    print("\nTop 10 Clustering Configurations (by Silhouette Score):")
    display_cols = ['Algorithm', 'Preprocessing', 'Parameters', 'Silhouette Score', 'n_clusters']
    print(results_df[display_cols].head(10).to_string(index=False))
    
    # Select the best configuration
    best_config = results_df.iloc[0]
    
    print("\n" + "=" * 60)
    print("BEST CLUSTERING CONFIGURATION SELECTED:")
    print("=" * 60)
    print(f"Algorithm: {best_config['Algorithm']}")
    print(f"Preprocessing: {best_config['Preprocessing']}")
    print(f"Parameters: {best_config['Parameters']}")
    print(f"Silhouette Score: {best_config['Silhouette Score']:.4f}")
    print(f"Number of Clusters: {best_config['n_clusters']}")
    
    # Return structured summary
    summary = {
        'algorithm': best_config['Algorithm'],
        'preprocessing': best_config['Preprocessing'],
        'parameters': best_config['Parameters'],
        'silhouette_score': best_config['Silhouette Score'],
        'n_clusters': best_config['n_clusters'],
        'labels': best_config['labels'],
        'all_results': results_df
    }
    
    return summary


# BONUS 1: SUBSPACE CLUSTERING VIA PCA

def bonus_subspace_clustering(data_dict):
    print("\n" + "-" * 80)
    print("BONUS 1: SUBSPACE CLUSTERING VIA PCA")
    
    X = data_dict['standardized']
    k = 3
    
    results = []
    n_components_range = range(2, 11)  # 2 to 10 components
    
    print(f"\nEvaluating K-Means (k={k}) across PCA subspaces...")
    
    for n_components in n_components_range:
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        
        # Compute metrics
        silhouette = silhouette_score(X_pca, labels)
        variance_explained = sum(pca.explained_variance_ratio_)
        
        results.append({
            'n_components': n_components,
            'Silhouette Score': silhouette,
            'Variance Explained': variance_explained,
            'Inertia': kmeans.inertia_
        })
    
    results_df = pd.DataFrame(results)
    print("\nResults by Number of PCA Components:")
    print(results_df.to_string(index=False))
    
    # Find optimal dimensionality
    best_idx = results_df['Silhouette Score'].idxmax()
    best_n_components = results_df.loc[best_idx, 'n_components']
    best_silhouette = results_df.loc[best_idx, 'Silhouette Score']
    
    print(f"\nOptimal dimensionality: {best_n_components} components")
    print(f"Best Silhouette Score: {best_silhouette:.4f}")
    
    return results_df, best_n_components


# BONUS 2: CLUSTER STABILITY UNDER NOISE

def bonus_cluster_stability_noise(data_dict):
    print("\n" + "-" * 80)
    print("BONUS 2: CLUSTER STABILITY UNDER NOISE")
    
    X = data_dict['standardized']
    k = 3
    noise_levels = [0.01, 0.05, 0.1]
    n_trials = 10
    
    # Get baseline clustering
    kmeans_baseline = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels_baseline = kmeans_baseline.fit_predict(X)
    
    results = []
    
    print(f"\nEvaluating cluster stability with {n_trials} trials per noise level...")
    
    for sigma in noise_levels:
        ari_scores = []
        
        for trial in range(n_trials):
            # Set different seed for each trial
            np.random.seed(RANDOM_SEED + trial)
            
            # Add Gaussian noise
            noise = np.random.normal(0, sigma, X.shape)
            X_noisy = X + noise
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels_noisy = kmeans.fit_predict(X_noisy)
            
            # Compute ARI with baseline
            ari = adjusted_rand_score(labels_baseline, labels_noisy)
            ari_scores.append(ari)
        
        results.append({
            'Noise Level (σ)': sigma,
            'Mean ARI': np.mean(ari_scores),
            'Std ARI': np.std(ari_scores),
            'Min ARI': np.min(ari_scores),
            'Max ARI': np.max(ari_scores)
        })
    
    results_df = pd.DataFrame(results)
    print("\nCluster Stability Results (Adjusted Rand Index):")
    print(results_df.to_string(index=False))
    
    # Analysis
    print("\nStability Analysis:")
    for _, row in results_df.iterrows():
        mean_ari = row['Mean ARI']
        sigma = row['Noise Level (σ)']
        if mean_ari > 0.9:
            stability = "Very stable"
        elif mean_ari > 0.7:
            stability = "Moderately stable"
        elif mean_ari > 0.5:
            stability = "Low stability"
        else:
            stability = "Unstable"
        print(f"   σ={sigma}: {stability} (Mean ARI = {mean_ari:.4f})")
    
    return results_df


# MAIN EXECUTION

def main():

    print("-" * 80)
    print("DATA MINING HOMEWORK 3: CLUSTERING ANALYSIS")
    print("Wine Dataset - Comprehensive Clustering Study")
    
    # Section 4: Data Preprocessing
    data_dict = load_and_preprocess_data()
    
    # Section 5: EDA and Visualization
    pca_data = perform_eda(data_dict, save_plots=True)
    
    # Part 1: Distance Behavior
    distance_results = analyze_distance_behavior(data_dict)
    
    # Part 2: Impact of Feature Scaling on K-Means
    scaling_results = analyze_scaling_impact(data_dict, k=3)
    
    # Part 3: K-Means from Scratch
    kmeans_comparison = compare_kmeans_implementations(data_dict, k=3)
    
    # Part 4: Hierarchical Clustering Stability
    hier_results, best_linkage = analyze_hierarchical_clustering(data_dict)
    
    # Part 5: Automated DBSCAN
    dbscan_results = automated_dbscan_clustering(data_dict)
    
    # Part 6: Fully Automated Clustering System
    auto_clustering_summary = automated_clustering_system(data_dict)
    
    # Bonus 1: Subspace Clustering via PCA
    subspace_results, optimal_dims = bonus_subspace_clustering(data_dict)
    
    # Bonus 2: Cluster Stability under Noise
    stability_results = bonus_cluster_stability_noise(data_dict)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE - ALL PARTS SUCCESSFULLY COMPLETED")
    print("=" * 80)
    print("\nGenerated Output Files:")
    print("   - eda_histograms.png")
    print("   - eda_correlation_matrix.png")
    print("   - eda_scatter_plots.png")
    print("   - eda_pca_visualization.png")
    print("\nAll results have been printed to the console above.")
    
    return {
        'data': data_dict,
        'distance_results': distance_results,
        'scaling_results': scaling_results,
        'kmeans_comparison': kmeans_comparison,
        'hierarchical_results': hier_results,
        'dbscan_results': dbscan_results,
        'auto_clustering': auto_clustering_summary,
        'subspace_results': subspace_results,
        'stability_results': stability_results
    }


if __name__ == "__main__":
    results = main()
