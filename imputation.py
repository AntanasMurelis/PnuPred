import numpy as np
import pandas as pd

def chk_missing(df):
    """Returns True/False whether dataframe contains missing values

    Args:
        df (DataFrame): DataFrame of interest
    Returns:
        Bool: True/False whether df contains missing values
    """
    return df.isnull().values.any()

def mean_impute(df, col, threshold = 0):
    """The DataFrame below the threshold in the column imputed with column mean. 

    Args:
        df (DataFrame): Dataframe to be imputed
        column (string): Column name
        threshold (int, optional): Threshold of replacement. Defaults to 0.

    Returns:
        DataFrame: The DataFrame below the threshold in the column imputed with column mean. 
    """
    
    imp = df.loc[df[col] > threshold, col]
    mean = imp.sum()/imp.count()
    df.loc[df[col] <= threshold, col] = mean
    
    return df

def value_cap(df, col, threshold):
    """Cap the column values with imputed mean 

    Args:
        df (DataFrame): DataFrame of interest
        col (string): column name
        threshold (real): Maximal value

    Returns:
        DataFrame: The DataFrame below the threshold in the column imputed with threshold value. 
    """
    imp = df.loc[df[col] < threshold, col]
    mean = imp.sum()/imp.count()
    df.loc[df[col] >= threshold, col] = threshold

    return df
    