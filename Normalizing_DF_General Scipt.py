# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:56:14 2024

@author: User
"""

#HOMEWORK 7.5
#Write a function to convert any dataframe into its normalized version, 
#with optional inputs of percentile of 5/95, standard, and min-max scalers.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalizing_data(df, method):
    #only work with numeric cols
    numeric_df=df.select_dtypes(include=[np.number])
    normalized_df=numeric_df.copy()

    if method == 'percentile':
        lower_percentile=np.percentile(numeric_df,5,axis=0)
        upper_percentile=np.percentile(numeric_df,95,axis=0)
        #Values below the 5th percentile are 0. 
        #Values above the 95th percentile are 1.
        normalized_df=np.clip(numeric_df, lower_percentile, upper_percentile)
        normalized_df=(normalized_df-lower_percentile)/(upper_percentile-lower_percentile)
        #Values within the 5th–95th percentile range are normalized to [0, 1]
    elif method == 'standard':
        scaler=StandardScaler()
        normalized_df=scaler.fit_transform(numeric_df)
    elif method == 'minmax':
        scaler=MinMaxScaler()
        normalized_df=scaler.fit_transform(numeric_df)
    else:
        raise ValueError("Invalid method. Choose 'percentile', 'standard', or 'minmax'.")
    # Convert back to DataFrame with original columns and index for standard and minmax case
    return pd.DataFrame(normalized_df,columns=numeric_df.columns, index=numeric_df.index)
#TESTING
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

normalized_percentile = normalizing_data(df, method="percentile")
print("Percentile Normalized:\n", normalized_percentile)


normalized_standard = normalizing_data(df, method="standard")
print("\nStandard Normalized:\n", normalized_standard)

normalized_minmax = normalizing_data(df, method="minmax")
print("\nMin-Max Normalized:\n", normalized_minmax)

