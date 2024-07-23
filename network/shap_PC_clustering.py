import pandas as pd
import numpy as np
from .custom_model import custom_model
import shap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

def get_elbow(final_model_custom: custom_model, train_X: pd.DataFrame) -> np.ndarray:
    """
    Calculate the distances for the elbow method using SHAP values and PCA components.

    Args:
        final_model_custom (custom_model): The trained custom model.
        train_X (pd.DataFrame): The training features.

    Returns:
        np.ndarray: Array of distances for the elbow method.
    """
    try:
        # Perform PCA transformation
        train_X_transformed = final_model_custom.pca_transform(train_X)

        # Calculate SHAP values
        explainer_custom = shap.TreeExplainer(final_model_custom.xgb_model, train_X_transformed)
        shap_values_custom = explainer_custom.shap_values(train_X_transformed)
        shap_values_custom_abs = np.abs(shap_values_custom)
        
        # Get PCA loadings
        loadings = final_model_custom.pca_model.components_
        num_principal_components = loadings.shape[0]
        pc_name_list = ["PC" + str(i + 1) for i in range(num_principal_components)]
        shap_df_custom_abs2 = pd.DataFrame(shap_values_custom_abs, columns=pc_name_list)

        # Calculate distances for the elbow method
        pcshap = shap_df_custom_abs2["PC1"].copy()
        pcshap = pcshap.values.reshape(-1, 1)
        pcloading = loadings[0, :].copy()
        pcloading = pcloading.reshape(1, -1)
        PC_sf_shap = np.dot(pcshap, pcloading)
        Z = linkage(PC_sf_shap.T, method='ward')
        distances = Z[:, 2][::-1]
        
        return distances
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating the elbow method: {e}")

def get_adj_matrix(final_model_custom: custom_model, train_X: pd.DataFrame, num_clusters: int = 10) -> pd.DataFrame:
    """
    Construct an adjacency matrix based on SHAP values and PCA components.

    Args:
        final_model_custom (custom_model): The trained custom model.
        train_X (pd.DataFrame): The training features.
        num_clusters (int): Number of clusters for hierarchical clustering.

    Returns:
        pd.DataFrame: Adjacency matrix showing feature connections.
    """
    try:
        # Perform PCA transformation
        train_X_transformed = final_model_custom.pca_transform(train_X)

        # Calculate SHAP values
        explainer_custom = shap.TreeExplainer(final_model_custom.xgb_model, train_X_transformed)
        shap_values_custom = explainer_custom.shap_values(train_X_transformed)
        shap_values_custom_abs = np.abs(shap_values_custom)
        
        # Get PCA loadings
        loadings = final_model_custom.pca_model.components_
        num_principal_components = loadings.shape[0]
        pc_name_list = ["PC" + str(i + 1) for i in range(num_principal_components)]
        shap_df_custom_abs2 = pd.DataFrame(shap_values_custom_abs, columns=pc_name_list)
        
        # Initialize adjacency matrix
        adj_matrix_whole_df = pd.DataFrame(np.zeros((train_X.shape[1], train_X.shape[1])), index=train_X.columns, columns=train_X.columns)

        for i, pci in enumerate(pc_name_list):
            # Calculate SHAP values for each PC
            pcshap = shap_df_custom_abs2[pci].copy()
            pcshap = pcshap.values.reshape(-1, 1)
            pcloading = loadings[i, :].copy()
            pcloading = pcloading.reshape(1, -1)
            PC_sf_shap = np.dot(pcshap, pcloading)
            PC_sf_shap_df = pd.DataFrame(PC_sf_shap, columns=train_X.columns)
            
            # Perform hierarchical clustering
            Z = linkage(PC_sf_shap.T, method='ward')
            dendrogram(Z, labels=PC_sf_shap_df.columns.tolist())
            plt.title(f'Dendrogram for PC {pci}')
            plt.xlabel('Features')
            plt.ylabel('Distance')
            plt.show()  # Show dendrogram plot

            # Determine clusters
            max_cluster_threshold = num_clusters + 1 
            cluster_labels = fcluster(Z, max_cluster_threshold, criterion='maxclust')
            clustered_data = PC_sf_shap_df.copy()
            clustered_data.loc['Cluster'] = cluster_labels
            clustered_data_pc = clustered_data.loc['Cluster']
            cluster_list = [clustered_data_pc.index[clustered_data_pc == clusterid].tolist() for clusterid in clustered_data_pc.unique()]
            
            # Update adjacency matrix
            for clist in cluster_list:
                for sf1 in clist:
                    for sf2 in clist:
                        if sf1 != sf2:
                            adj_matrix_whole_df.loc[sf1, sf2] += 1
        
        return adj_matrix_whole_df
    except Exception as e:
        raise RuntimeError(f"An error occurred while constructing the adjacency matrix: {e}")
