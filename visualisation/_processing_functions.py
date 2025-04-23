import pandas as pd

def make_callsign_column(dataframe):
    dataframe['callsign'] = dataframe['vehicle_type'].str[0].str.upper() + dataframe['callsign_group'].fillna(0).astype(int).astype(str)
    return dataframe

def calculate_time_difference(df, col1, col2, unit='minutes'):
    """
    Calculate the time difference between two datetime columns in a Pandas DataFrame.
    *AI PRODUCED FUNCTION - CHECKED FOR CORRECT FUNCTIONING*

    Args:
        df (pd.DataFrame): The DataFrame containing the datetime columns.
        col1 (str): Name of the first datetime column.
        col2 (str): Name of the second datetime column.
        unit (str): The unit for the time difference ('seconds', 'minutes', 'hours', 'days').

    Returns:
        pd.Series: A Pandas Series with the time differences in the specified unit.
    """
    # Convert columns to datetime format
    df[col1] = pd.to_datetime(df[col1], format='ISO8601')
    df[col2] = pd.to_datetime(df[col2], format='ISO8601')

    # Compute time difference
    time_diff = df[col2] - df[col1]

    # Convert to specified unit
    if unit == 'seconds':
        return time_diff.dt.total_seconds()
    elif unit == 'minutes':
        return time_diff.dt.total_seconds() / 60
    elif unit == 'hours':
        return time_diff.dt.total_seconds() / 3600
    elif unit == 'days':
        return time_diff.dt.total_seconds() / 86400
    else:
        raise ValueError("Invalid unit. Choose from 'seconds', 'minutes', 'hours', or 'days'.")

def get_param(parameter, params_df):
    return params_df[params_df["parameter"] == parameter]['value'].values[0]

def fill_missing_values(df, column, value):
    """
    Replaces missing (NaN) values in a specified column with a given value.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        column (str): The name of the column to process.
        value (any): The value to replace NaN values with.

    Returns:
        pd.DataFrame: The modified DataFrame with NaNs replaced.
    """
    df[column] = df[column].fillna(value)
    return df
