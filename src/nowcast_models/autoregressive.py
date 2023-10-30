import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.stattools import adfuller


def select_optimal_lag(data, max_lag=10, information_criteria="aic"):
    """
    Select the optimal lag order based on the specified information criteria.

    Parameters:
        data (pd.Series): The time series data.
        max_lag (int): The maximum number of lags to consider.
        information_criteria (str): The information criteria to use for lag selection.

    Returns:
        list[int]: The selected number of lags.
    """
    sel = ar_select_order(data, maxlag=max_lag, ic=information_criteria)
    return sel.ar_lags


def split_estimate_forecast(data, train_end_date, forecast_end_date):
    """
    Split the data into estimation and forecast datasets based on specified dates.

    Parameters:
        data (pd.DataFrame): The time series data.
        train_end_date (str): The end date for the estimation dataset.
        forecast_end_date (str): The end date for the forecast dataset.

    Returns:
        pd.DataFrame: The estimation dataset.
        pd.DataFrame: The forecast dataset.
    """
    estimate_df = data[data.index <= train_end_date]
    forecast_df = data[
        (data.index > train_end_date) & (data.index <= forecast_end_date)
    ]
    return estimate_df, forecast_df


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


def apply_autoregression(data, config, max_lag=10):
    """
    Apply the autoregressive model to the data.

    Parameters:
        data (pd.Series or pd.DataFrame): The time series data.
        config (dict): The configuration dictionary.
        max_lag (int): The maximum number of lags to consider.

    Returns:
        pd.DataFrame: The training data.
        pd.DataFrame: The predictions from the autoregressive model along with confidence intervals and test data.
    """
    # Ensure data is a pandas Series
    if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
        data = data.squeeze()  # Convert single-column DataFrame to Series
    elif not isinstance(data, pd.Series):
        raise ValueError(
            "Input data should be a pandas Series or single-column DataFrame"
        )

    # Define the end dates for estimation and forecast datasets
    end_date_estimate = config.get(
        "end_date_estimate", "2014-04-01"
    )  # default to "2014-04-01" if not found in config
    end_date_outcome = config.get(
        "end_date_outcome", "2019-10-01"
    )  # default to "2019-10-01" if not found in config

    # Compute quarterly percentage change
    data_pct_change = data.pct_change().dropna()

    # Split the data
    train_data, test_data = split_estimate_forecast(
        data_pct_change, end_date_estimate, end_date_outcome
    )

    # Check for stationarity
    _, is_stationary = check_stationarity(train_data)

    # If not stationary, transform the data
    if not is_stationary:
        logging.info("Data is not stationary. Applying transformations...")
        # Combine, transform, and split again to keep consistent starting points
        combined_data = pd.concat([train_data, test_data])
        transformed_data = transform_data(
            combined_data, transformations=["differencing"]
        )
        train_data, test_data = split_estimate_forecast(
            transformed_data, end_date_estimate, end_date_outcome
        )

    # Check for stationarity again
    _, is_stationary = check_stationarity(train_data)
    if not is_stationary:
        logging.warning(
            "Data is still not stationary after transforming. Inspect before continuing."
        )

    # Select the optimal lag
    opt_lag = select_optimal_lag(train_data, max_lag=max_lag)

    # Fit the autoregressive model
    mod = AutoReg(train_data, opt_lag).fit()

    # Get predictions along with confidence intervals
    pred_obj = mod.get_prediction(
        start=test_data.index.min(), end=test_data.index.max()
    )
    preds = pred_obj.predicted_mean
    conf_int = pred_obj.conf_int()

    # Prepare the prediction DataFrame
    ar_prediction = pd.DataFrame(
        {
            "Actual": test_data,
            "Prediction": preds,
            "Lower CI": conf_int.iloc[:, 0],
            "Upper CI": conf_int.iloc[:, 1],
        }
    )

    return train_data, ar_prediction
