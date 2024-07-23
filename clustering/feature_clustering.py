import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from typing import Dict, Tuple

def feature_clustering(adj_matrix_whole_df: pd.DataFrame, num_clusters: int = 10) -> Tuple[Dict[int, list], pd.DataFrame]:
    """
    Perform spectral clustering on the adjacency matrix and return clustered gene lists.

    Args:
        adj_matrix_whole_df (pd.DataFrame): The adjacency matrix where rows and columns represent genes.
        num_clusters (int): The number of clusters to form. Default is 10.

    Returns:
        Tuple[Dict[int, list], pd.DataFrame]: A tuple where the first element is a dictionary
            with cluster identifiers as keys and lists of gene names as values, and the second element
            is the reordered adjacency matrix DataFrame.
    """
    try:
        adj_matrix = adj_matrix_whole_df.values

        # Perform spectral clustering on rows (genes)
        spectral_row = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
        row_clusters = spectral_row.fit_predict(adj_matrix)

        # Perform spectral clustering on columns
        spectral_col = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
        col_clusters = spectral_col.fit_predict(adj_matrix.T)

        # Reorder the adjacency matrix
        row_order = np.argsort(row_clusters)
        col_order = np.argsort(col_clusters)
        ordered_adj_matrix = adj_matrix[row_order, :][:, col_order]

        # Create a DataFrame with the reordered matrix and appropriate labels
        ordered_adj_matrix_df = pd.DataFrame(ordered_adj_matrix, index=adj_matrix_whole_df.index[row_order], columns=adj_matrix_whole_df.columns[col_order])

        # Convert column clusters to a Series with appropriate index
        col_cluster_series = pd.Series(col_clusters, index=adj_matrix_whole_df.columns)

        # Group genes by their cluster assignments
        clustered_genes = {cluster_id: col_cluster_series.index[col_cluster_series == cluster_id].tolist() for cluster_id in col_cluster_series.unique()}

        return clustered_genes, ordered_adj_matrix_df

    except Exception as e:
        raise RuntimeError(f"An error occurred while performing feature clustering: {e}")

def spectral_elbow(adj_matrix_whole_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the sorted eigenvalues of the Laplacian matrix for the adjacency matrix.

    Args:
        adj_matrix_whole_df (pd.DataFrame): The adjacency matrix where rows and columns represent genes.

    Returns:
        np.ndarray: Sorted eigenvalues of the Laplacian matrix.
    """
    try:
        adj_matrix = adj_matrix_whole_df.values

        # Compute the Laplacian matrix
        L = laplacian(adj_matrix, normed=True)

        # Compute eigenvalues of the Laplacian matrix
        eigenvalues, _ = eigh(L)

        # Sort eigenvalues in ascending order
        sorted_eigenvalues = np.sort(eigenvalues)

        return sorted_eigenvalues

    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating the spectral elbow: {e}")
