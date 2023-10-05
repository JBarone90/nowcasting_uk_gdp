import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def get_optimal_lags(data):
    """
    Assigns optimal number of lags for the given data using ADF test.

    Parameters
    ----------
    data : Pandas Series
        Time series data.

    Returns
    -------
    optimal_lag : int
        Optimal number of lags.
    """
    result = adfuller(data.dropna(), autolag="AIC")
    optimal_lag = result[2]
    return max(optimal_lag, 1)


def create_lagged_features(data, lags):
    """
    Creates lagged features for the given data.

    Parameters
    ----------
    data : Pandas Series
        Time series data.
    lags : int
        Number of lags to create.

    Returns
    -------
    lagged_data : Pandas DataFrame
        DataFrame with lagged features.
    """
    lagged_data = pd.DataFrame(data)
    for lag in range(1, lags + 1):
        lagged_data[f"Lag{lag}"] = data.shift(lag)
    return lagged_data.dropna()


def split_data(data, start, end):
    """
    Splits the data into estimation and forecast sets based on the time range.

    Parameters
    ----------
    data : Pandas DataFrame
        Data containing the target variable and lagged features.
    start : str
        Start date of the estimation period (e.g., "2005-04").
    end : str
        End date of the estimation period (e.g., "2019-10").

    Returns
    -------
    estimation_data : Pandas DataFrame
        Data for model estimation.
    forecast_data : Pandas DataFrame
        Data for making forecasts.
    """
    estimation_data = data[(data.index >= start) & (data.index < end)]
    forecast_data = data[data.index >= end]
    return estimation_data, forecast_data


def estimate_autoregressive_model(estimation_data, target_col):
    """
    Estimates the autoregressive model using OLS.

    Parameters
    ----------
    estimation_data : Pandas DataFrame
        Data for model estimation (with lagged features).
    target_col : str
        Name of the target variable column.

    Returns
    -------
    model_results : Statsmodels model fit
        Fit results of the autoregressive model.
    """
    X = estimation_data.drop(target_col, axis=1)
    X = sm.add_constant(X)  # Add a constant for the intercept term
    y = estimation_data[target_col]
    model = sm.OLS(y, X)
    model_results = model.fit()
    return model_results


def make_forecasts(model, forecast_data, lags, target_col):
    """
    Makes forecasts for the given forecast data using the autoregressive model.

    Parameters
    ----------
    model : Statsmodels model fit
        Fitted autoregressive model.
    forecast_data : Pandas DataFrame
        Data for making forecasts (with lagged features).
    lags : int
        Number of lags used in the model.
    target_col : str
        Name of the target variable column.

    Returns
    -------
    forecasts : Pandas Series
        Predicted values for the forecast period.
    """
    forecast_data = forecast_data.tail(lags)  # Select the last lags rows for prediction
    X_forecast = forecast_data.drop(target_col, axis=1)
    X_forecast = sm.add_constant(X_forecast)
    forecasts = model.predict(X_forecast)
    return forecasts
