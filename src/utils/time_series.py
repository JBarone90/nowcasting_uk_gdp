import numpy as np
from statsmodels.tsa.stattools import adfuller


def check_stationarity(data):
    """
    Check if the time series data is stationary.

    Parameters:
        data (pd.Series): The time series data.

    Returns:
        dict: A dictionary containing the ADF statistic, p-value, and critical values.
        bool: A boolean indicating whether the data is stationary.
    """
    result = adfuller(data)
    output = {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4],
    }
    is_stationary = result[1] <= 0.05  # Assuming a common alpha of 0.05
    return output, is_stationary


def transform_data(data, transformations=["differencing"]):
    """
    Perform specified transformations on the time series data.

    Parameters:
        data (pd.Series): The time series data.
        transformations (list): A list of transformations to apply.
                                Available transformations: 'differencing', 'log', 'square_root'.

    Returns:
        pd.Series: The transformed data.
    """
    transformed_data = data.copy()
    for transformation in transformations:
        if transformation == "differencing":
            transformed_data = transformed_data.diff().dropna()
        elif transformation == "log":
            transformed_data = np.log(transformed_data).dropna()
        elif transformation == "square_root":
            transformed_data = np.sqrt(transformed_data).dropna()
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    return transformed_data
