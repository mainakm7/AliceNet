import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

def feature_clustering(adj_matrix_whole_df: pd.DataFrame, num_clusters: int = 10) -> dict:
    """
    Perform spectral clustering on the adjacency matrix and return clustered gene lists.

    Args:
        adj_matrix_whole_df (pd.DataFrame): The adjacency matrix where rows and columns represent genes.
        num_clusters (int): The number of clusters to form. Default is 10.

    Returns:
        dict: A dictionary where keys are cluster identifiers and values are lists of gene names in each cluster.
    """
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
    clustered_genes = [col_cluster_series.index[col_cluster_series == cluster_id].tolist() for cluster_id in col_cluster_series.unique()]

    return clustered_genes
