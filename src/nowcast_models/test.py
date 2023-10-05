import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score


def apply_lags(data, opt_lag, target="value"):
    """
    Adds additional columns to a dataframe for
    each variable for specified lags

    Parameters
    ----------
        data : Pandas Dataframe
            Dataframe of variables

    Returns
    -------
        data: Pandas Dataframe
            Dataframe with new lagged columns
    """

    for lag in range(1, opt_lag + 1):
        data[f"lag_{lag}"] = data[f"{target}"].shift(lag)

    data = data.sort_index(axis=1)

    return data.dropna()


def select_optimal_lag(data, max_lag=10):
    """
    Select the optimal lag order using the Akaike Information Criterion (AIC).

    Args:
        data (pd.DataFrame): Time series data with a 'value' column.
        max_lag (int): Maximum lag order to consider.

    Returns:
        int: Optimal lag order selected based on AIC.
    """
    best_aic = float("inf")
    best_lag = None

    for lag in range(1, max_lag + 1):
        # Add one lag at the time
        data.loc[:, f"lag_{lag}"] = data["value"].shift(lag)

        # Remove NaNs and assing it to a temporary dataframe
        tmp = data.dropna()
        y = tmp["value"]
        X = tmp[["lag_" + str(i) for i in range(1, lag + 1)]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        current_aic = model.aic

        if current_aic < best_aic:
            best_aic = current_aic
            best_lag = lag

    return best_lag


def fit_autoregressive_model(data, lag_order):
    """
    Fit an autoregressive model to the time series data.

    Args:
        data (pd.DataFrame): Time series data with a 'value' column.
        lag_order (int): Lag order for the autoregressive model.

    Returns:
        sm.OLS: Fitted autoregressive model.
    """
    for i in range(1, lag_order + 1):
        data[f"lag_{i}"] = data["value"].shift(i)

    data.dropna(inplace=True)
    y = data["value"]
    X = data[["lag_" + str(i) for i in range(1, lag_order + 1)]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def evaluate_model(model, data):
    """
    Evaluate the autoregressive model using validation data.

    Args:
        model (sm.OLS): Fitted autoregressive model.
        data (pd.DataFrame): Validation data with a 'value' column.

    Returns:
        float: Mean squared error (MSE) of the model on the validation data.
        float: R-squared (R2) of the model on the validation data.
    """
    y_validation = data["value"]
    X_validation = data[["lag_" + str(i) for i in range(1, model.params.size)]]
    X_validation = sm.add_constant(X_validation)
    y_pred = model.predict(X_validation)
    mse = mean_squared_error(y_validation, y_pred)
    r2 = r2_score(y_validation, y_pred)
    return mse, r2
