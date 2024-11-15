import numpy as np
import polars as pl
import os
from src.utils import get_project_root
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats

# Path to the parent directory containing folders with Parquet files
parent_directory = f'{get_project_root()}/train.parquet'

# List to store individual dataframes
dataframes = []
features = pd.read_csv(f'{get_project_root()}/features.csv', sep=',')
features = list(features['feature'])

# Iterate over each folder in the parent directory
for folder_name in os.listdir(parent_directory):
    if folder_name == 'partition_id=0':

        folder_path = os.path.join(parent_directory, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Look for parquet files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".parquet"):
                    file_path = os.path.join(folder_path, file_name)
                    print(file_path)
                    # Read the parquet file and append to the list
                    df = pl.read_parquet(file_path)
                    dataframes.append(df)
    else:
        pass
# Concatenate all dataframes into one
combined_df = pl.concat(dataframes)
to_remove = []
for i in range(0, 9):
    if i == 6:
        pass
    else:
        to_remove.append(f'responder_{i}')

combined_df = combined_df.drop(to_remove)
# Now `combined_df` contains data from all Parquet files
symbol_1 = combined_df.filter(pl.col('symbol_id') == 1)

symbol_1 = symbol_1.insert_column(0,
                                  pl.concat_str(
                                      [
                                          pl.col("date_id"),
                                          pl.col("time_id"),
                                      ],
                                      separator="_",
                                  ).alias("date_time_id"))
df_polars = symbol_1.drop(['date_id', 'time_id'])
not_null_cols = filter(lambda x: x.null_count() != df_polars.height, df_polars)
not_null_col_names = map(lambda x: x.name, not_null_cols)

not_unique_value_cols = filter(lambda x: x.n_unique() != 1, df_polars)
not_unique_value_col_names = map(lambda x: x.name, not_unique_value_cols)

df_polars = df_polars.select(not_null_col_names)
df_polars = df_polars.select(not_unique_value_col_names)

print(df_polars.select(pl.mean("responder_6")))
print(df_polars.select(pl.std("responder_6")))

df_pandas = pd.DataFrame(df_polars, columns=df_polars.columns)
sns.displot(df_pandas, x="responder_6", bins=100)
plt.show()

feats = df_pandas.filter(regex='feature').columns
