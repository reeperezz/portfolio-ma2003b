"""
Reusable utilities for the MegaMart Customer Segmentation project.

Provides functions for:
- Loading and summarizing the retail customer dataset.
- Standardizing behavioral variables before clustering.
- Running K-Means across a range of k values (for elbow and silhouette analysis).
- Fitting a final K-Means model and attaching cluster labels to the original data.
- Computing cluster profile tables (mean behavior per cluster).
- Running and visualizing PCA in 2D for cluster interpretation.
- Optional plotting helpers for elbow, silhouette, and PCA scatter plots.

These helpers mirror common steps in `retail.ipynb`.

Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy (for dendrograms, optional).
"""

from typing import Tuple, Dict, Optional, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.cluster.hierarchy import linkage, dendrogram  # optional for hierarchical plots
except Exception:
    linkage = None
    dendrogram = None



def load_data(csv_path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Load the retail CSV into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file.
        parse_dates: Optional list of column names to parse as dates.

    Returns:
        DataFrame with loaded data.
    """
    return pd.read_csv(csv_path, parse_dates=parse_dates)


def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with basic stats for numeric columns.

    Includes: count, mean, std, min, max, and percent missing.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame indexed by column name with summary statistics.
    """
    num = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame({
        "count": num.count(),
        "mean": num.mean(),
        "std": num.std(),
        "min": num.min(),
        "max": num.max(),
        "%missing": num.isna().mean() * 100,
    })
    return summary




def standardize_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str]
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize selected numeric features using StandardScaler.

    Args:
        df: Input DataFrame.
        feature_cols: Columns to standardize.

    Returns:
        X_scaled: Numpy array with standardized values.
        scaler: Fitted StandardScaler instance.
    """
    feature_cols = list(feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler




def compute_linkage(
    X: np.ndarray,
    method: str = "ward"
) -> Optional[np.ndarray]:
    """Compute a linkage matrix for hierarchical clustering.

    Args:
        X: Standardized feature matrix.
        method: Linkage method, e.g., 'ward', 'single', 'complete', 'average'.

    Returns:
        Linkage matrix (np.ndarray) or None if scipy is not installed.
    """
    if linkage is None:
        return None
    return linkage(X, method=method)


def plot_dendrogram(
    Z: np.ndarray,
    title: str = "Dendrogram",
    truncate_mode: Optional[str] = None,
    p: int = 20
) -> None:
    """Plot a dendrogram given a linkage matrix.

    Args:
        Z: Linkage matrix.
        title: Plot title.
        truncate_mode: Truncation mode for dendrogram (e.g., 'lastp').
        p: Number of last merged clusters to show if truncate_mode is 'lastp'.
    """
    if dendrogram is None:
        print("scipy is not available; cannot plot dendrogram.")
        return

    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode=truncate_mode, p=p)
    plt.title(title)
    plt.xlabel("Merged clusters / samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()




def run_kmeans_range(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
    n_init: int = 10
) -> pd.DataFrame:
    """Run K-Means for a range of k and return inertia and silhouette scores.

    Args:
        X: Standardized feature matrix.
        k_values: Iterable of k values to test (e.g., range(2, 11)).
        random_state: Random seed for KMeans.
        n_init: Number of initializations.

    Returns:
        DataFrame with columns: 'k', 'inertia', 'silhouette_score'.
    """
    results = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        model.fit(X)
        labels = model.labels_
        inertia = float(model.inertia_)
        sil = float(silhouette_score(X, labels))
        results.append({"k": k, "inertia": inertia, "silhouette_score": sil})

    return pd.DataFrame(results)


def plot_elbow_and_silhouette(results_df: pd.DataFrame) -> None:
    """Plot elbow (k vs inertia) and silhouette (k vs score) side by side.

    Args:
        results_df: DataFrame returned by run_kmeans_range.
    """
    if not {"k", "inertia", "silhouette_score"}.issubset(results_df.columns):
        raise ValueError("results_df must contain 'k', 'inertia', 'silhouette_score' columns.")

    k_vals = results_df["k"].values
    inertias = results_df["inertia"].values
    sils = results_df["silhouette_score"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_vals, inertias, "o-", color="blue")
    ax1.set_title("Elbow Method (k vs inertia)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.grid(True)

    ax2.plot(k_vals, sils, "o-", color="green")
    ax2.set_title("Silhouette Score vs k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()




def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10
) -> Dict:
    """Fit a K-Means model for a fixed number of clusters.

    Args:
        X: Standardized feature matrix.
        n_clusters: Number of clusters.
        random_state: Random seed.
        n_init: Number of initializations.

    Returns:
        Dict with keys: 'model', 'labels', 'inertia', 'silhouette'.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    model.fit(X)
    labels = model.labels_
    inertia = float(model.inertia_)
    sil = float(silhouette_score(X, labels))

    return {
        "model": model,
        "labels": labels,
        "inertia": inertia,
        "silhouette": sil
    }


def add_cluster_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_col: str = "cluster"
) -> pd.DataFrame:
    """Attach cluster labels to a copy of the original DataFrame.

    Args:
        df: Original DataFrame.
        labels: Cluster labels from K-Means (length must match df).
        cluster_col: Column name for the cluster labels.

    Returns:
        DataFrame with an added cluster column.
    """
    if len(labels) != len(df):
        raise ValueError("Length of labels must match number of rows in df.")
    df_out = df.copy()
    df_out[cluster_col] = labels
    return df_out


def cluster_profiles(
    df_with_clusters: pd.DataFrame,
    cluster_col: str,
    feature_cols: Iterable[str],
    round_decimals: int = 2
) -> pd.DataFrame:
    """Compute mean profiles for each cluster across selected features.

    Args:
        df_with_clusters: DataFrame that already contains a cluster label column.
        cluster_col: Name of the cluster label column.
        feature_cols: Numeric feature columns to profile.
        round_decimals: Number of decimals to round the means.

    Returns:
        DataFrame where rows are clusters and columns are feature means.
    """
    feature_cols = list(feature_cols)
    grouped = df_with_clusters.groupby(cluster_col)[feature_cols].mean()
    return grouped.round(round_decimals)




def plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Silhouette Plot for K-Means"
) -> None:
    """Create a silhouette plot for a fitted K-Means configuration.

    Args:
        X: Standardized feature matrix.
        labels: Cluster labels.
        title: Plot title.
    """
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(X, labels)
    y_lower = 10

    plt.figure(figsize=(10, 6))

    for cluster_id in range(n_clusters):
        cluster_vals = silhouette_vals[labels == cluster_id]
        cluster_vals.sort()
        size_cluster = cluster_vals.shape[0]
        y_upper = y_lower + size_cluster

        color = plt.cm.tab10(cluster_id / max(1, n_clusters - 1))
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
        y_lower = y_upper + 10

    avg_sil = silhouette_score(X, labels)
    plt.axvline(x=avg_sil, color="red", linestyle="--", label=f"Average: {avg_sil:.3f}")
    plt.title(title)
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.legend()
    plt.tight_layout()
    plt.show()




def run_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[PCA, np.ndarray]:
    """Run PCA on the standardized feature matrix.

    Args:
        X: Standardized feature matrix.
        n_components: Number of principal components.
        random_state: Random seed.

    Returns:
        pca: Fitted PCA object.
        X_pca: Transformed coordinates (n_samples x n_components).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def make_pca_dataframe(
    X_pca: np.ndarray,
    labels: Optional[np.ndarray] = None,
    cluster_col: str = "cluster"
) -> pd.DataFrame:
    """Create a DataFrame for PCA coordinates, optionally with cluster labels.

    Args:
        X_pca: PCA coordinates (n_samples x 2).
        labels: Optional cluster labels.
        cluster_col: Name of the cluster label column.

    Returns:
        DataFrame with columns ['PC1', 'PC2'] and optional cluster column.
    """
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    if labels is not None:
        if len(labels) != len(df_pca):
            raise ValueError("Length of labels must match number of rows in PCA data.")
        df_pca[cluster_col] = labels
    return df_pca


def plot_pca_clusters(
    df_pca: pd.DataFrame,
    cluster_col: str = "cluster",
    centroids_pca: Optional[np.ndarray] = None,
    title: str = "K-Means Clusters in PCA Space"
) -> None:
    """Plot PCA scatter of customers colored by cluster, with optional centroids.

    Args:
        df_pca: DataFrame with PC1, PC2 and a cluster column.
        cluster_col: Name of the cluster label column.
        centroids_pca: Optional array of centroid coordinates in PCA space.
        title: Plot title.
    """
    if not {"PC1", "PC2"}.issubset(df_pca.columns):
        raise ValueError("df_pca must contain 'PC1' and 'PC2' columns.")

    plt.figure(figsize=(10, 7))
    unique_clusters = sorted(df_pca[cluster_col].unique())

    for cluster_id in unique_clusters:
        subset = df_pca[df_pca[cluster_col] == cluster_id]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            s=30,
            alpha=0.7,
            label=f"Cluster {cluster_id}"
        )

    if centroids_pca is not None:
        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            s=200,
            c="black",
            marker="X",
            label="Centroids"
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()




def export_dataframe(df: pd.DataFrame, path: str, index: bool = True) -> None:
    """Save a DataFrame to CSV.

    Args:
        df: DataFrame to export.
        path: Output path (CSV).
        index: Whether to include the index in the file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index)


if __name__ == "__main__":
    # Simple smoke test (does not run any heavy computations).
    print("utils.py: helper module for MegaMart customer segmentation project.")
