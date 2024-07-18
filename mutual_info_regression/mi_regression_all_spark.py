from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col
from pyspark.sql.types import FloatType
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from datetime import datetime
import os
from ..utils.data_loader import sf_events_upd, sf_exp_upd
from ..utils.data_dir_path import data_dir_path

def compute_mutual_info(gene_event_tuple):
    """
    Compute mutual information between a gene and a splicing event.

    Args:
        gene_event_tuple (tuple): Tuple containing gene and event values.

    Returns:
        float: Mutual information regression value.
    """
    gene, event = gene_event_tuple
    # Filter out NaNs
    mask = ~np.isnan(event)
    y = event[mask]
    X = gene[mask].reshape(-1, 1)

    # Compute mutual information regression
    if len(y) > 0:  # Ensure there's data to compute
        mi_reg_val = mutual_info_regression(X, y)
        return float(mi_reg_val[0])
    else:
        return float('nan')

def mi_regression_all():
    """
    Perform mutual information regression for all combinations of genes and splicing events,
    save results to a CSV file.

    Returns:
        Spark DataFrame: DataFrame containing mutual information regression values.
    """
    # Create a Spark session
    spark = SparkSession.builder.appName("MIRegression").getOrCreate()

    # Convert pandas DataFrames to Spark DataFrames
    sf_events_sdf = spark.createDataFrame(sf_events_upd.T.reset_index())
    sf_exp_sdf = spark.createDataFrame(sf_exp_upd.T.reset_index())

    # Create cartesian product of the two dataframes
    cartesian_df = sf_exp_sdf.crossJoin(sf_events_sdf)

    # Create gene-event tuples
    gene_event_rdd = cartesian_df.rdd.map(lambda row: (
        np.array([row[col] for col in sf_exp_upd.columns], dtype=float),
        np.array([row[col] for col in sf_events_upd.columns], dtype=float)
    ))

    # Compute mutual information for each tuple
    mi_values_rdd = gene_event_rdd.map(compute_mutual_info)

    # Add mutual information values back to the dataframe
    mi_values_sdf = cartesian_df.withColumn("mi", mi_values_rdd.map(lambda x: float(x)).toDF("mi").mi.cast(FloatType()))

    # Pivot the dataframe to get the desired format
    mi_reg_df = mi_values_sdf.groupBy("index_x").pivot("index_y").agg({"mi": "first"})

    # Define the save path using current timestamp
    data_path_whole = data_dir_path()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(data_path_whole, f"mi_reg_all_{timestamp}.csv")

    # Save to CSV
    mi_reg_df.coalesce(1).write.csv(save_path, header=True)

    return mi_reg_df

# Call the function
mi_regression_all()
