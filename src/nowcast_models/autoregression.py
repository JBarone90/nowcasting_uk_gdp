import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def get_optimal_lags(df):
    """
    Assigns optimal number of lags for each variable given
    in dataframe

    Parameters
    ----------
        df : Pandas Dataframe
            Data passed into the method

    Returns
    -------
        lagn: list[int]
            list of lags for respective variables
    """
    lagn = []
    for k in df.columns:
        lags = assign_optimum_lags(df[k])
        lagn.append(max(lags, 1))
    return lagn


def assign_optimum_lags(input_df):
    """
    Assesses variable given via dataframe using
    A-DF test for lag assignment

    Parameters
    ----------
        df : Pandas Dataframe
            Data passed into the method

    Returns
    -------
        opt_lag_n: int
            lags assigned to given variable
    """
    opt_lag = adfuller(input_df.dropna(), autolag="AIC")
    opt_lag_n = opt_lag[2]

    # to avoid nonsensical lag selections, impose an arbitrary cutoff value
    if opt_lag_n > 5:
        opt_lag_n = 5
        return opt_lag_n
    else:
        return opt_lag_n


def split_data(df, end_estimation, model_params):
    """
    Subsets data into the dataframes for the outcome data and the explanatory variables

    Parameters
    ----------
        df : Pandas Dataframe
            All data passed into the method
        end_estimation : datetime date
            last date/month in the estimation period
        model_params:
            model params from the config file

    Returns
    -------
        estimate_data: Pandas Dataframe
            Dataframe of explanatory variables
        outcome_data: Pandas Dataframe
            Dataframe with outcome variable only
    """

    estimate_data = df[df.index <= end_estimation].drop(
        columns=[model_params["target"]]
    )

    outcome_data = df[[model_params["target"]]].dropna()

    return estimate_data, outcome_data


def get_variable_periods(df, model_params):
    """
    Returns dates for the end of estimation period and forecasting quarter

    Parameters
    ----------
        df : Pandas Dataframe
            All data passed into the method
        model_params:
            model params from the config file

    Returns
    -------
        nowcast_period: String(Datetime)
            Forecast quarter define by the third month in the quarter
        start_estimation: Datetime
            Beginning of the estimation period
        end_estimation: String(Datetime)
            end of the estimation period
    """

    nowcast_period = str(df.index.max().date())
    start_estimation = model_params["start"]
    end_estimation = str(df.iloc[:-3,].index.max().date())

    return nowcast_period, start_estimation, end_estimation


def shift_dataframe(df, shift_n):
    """
    Shifts dataframe to align realisation to predictors in preparation
    for OLS estimate model

    Parameters
    ----------
        df : Pandas Dataframe
            Dataframe of variables
        shift_n:
            number of rows to shift dataframe by

    Returns
    -------
        df: Pandas Dataframe
            Dataframe with shifted values
    """
    df_copy = df.copy()

    df_copy["ref_date"] = df_copy.index
    df_copy["ref_date_shift"] = df_copy.ref_date.shift(shift_n)

    df = df[df.index.notnull()]

    return df


def get_low_freq_rows(df):
    """
    Returns latest quarterly values from dataframe passed
    in with monthly index

    Parameters
    ----------
        df : Pandas Dataframe
            Dataframe of variables

    Returns
    -------
        df: Pandas Dataframe
            Dataframe subsetted for quarterly rows
    """
    df = df[df.index.month % 3 == 0]
    return df


def apply_lags(df, lags):
    """
    Adds additional columns to a dataframe for
    each variable for specified lags

    Parameters
    ----------
        df : Pandas Dataframe
            Dataframe of variables
        lags: List[int]
            List of lags to be applied to each variable

    Returns
    -------
        df: Pandas Dataframe
            Dataframe with new lagged columns
    """

    for column in df.columns:
        lags_for_current_var = lags[df.columns.get_loc(column)]
        for n in range(1, lags_for_current_var + 1):
            df[f"{column}_lag{n}"] = df[f"{column}"].shift(n)

    df = df.sort_index(axis=1)

    return df.dropna()


def get_lagged_outcome(df, lags):
    """
    Creates new lagged columns for outcome variable given specified lags

    Parameters
    ----------
        df : Pandas Dataframe
            Dataframe of variables
        lags: List[int]
            List of lags to be applied to each variable

    Returns
    -------
        df: Pandas Dataframe
            Dataframe with new lagged columns
    """

    column = str(df.columns[0])

    for n in range(1, lags + 1):
        df[f"{column}_lag{n}"] = df[f"{column}"].shift(n)

    return df


def combine_data(df_explantory, df_outcome):
    """
    Combines explanatory and outcome (with lags) dataframes

    Parameters
    ----------
        df_explanatory : Pandas Dataframe
            Dataframe of variables
        df_outcome : Pandas Dataframe
            Dataframe of variables

    Returns
    -------
        merged_data: Pandas Dataframe
            merged_data
    """

    merged_data = df_outcome.merge(
        df_explantory, left_index=True, right_index=True, how="outer"
    )
    # shift outcome data
    for column in df_explantory.columns:
        merged_data[f"{column}"] = merged_data[column].shift(1)

    return merged_data


def split_estimate_forecast(df, forecast_period, target):
    """
    Splits organised data into target and explanatory variables set for
    estimate and forecast models

    Parameters
    ----------
        df : Pandas Dataframe
            Dataframe of all variables
        forecast period : Pandas Dataframe
            Dataframe of variables
        target: String
            Target(outcome variable)

    Returns
    -------
        x_estimate_values: Pandas Dataframe
            explantory variables dataframe for estimate
        y_estimate_values: Pandas Dataframe
            outcome variable dataframe for estimate
        x_forecast_values: Pandas Dataframe
            value to use in forecasting next period
    """

    df_estimate_set = df[df.index < forecast_period]
    df_estimate_set = df_estimate_set.dropna()
    df_forecast_set = df[df.index == forecast_period]

    x_estimate_values = df_estimate_set.drop(target, axis=1)
    y_estimate_values = df_estimate_set[[target]]

    x_forecast_values = df_forecast_set.drop(target, axis=1)
    return x_estimate_values, y_estimate_values, x_forecast_values


def get_autoregression_estimate(x_values, y_value):
    """
    Produces OLS estimates

    Parameters
    ----------
        x_values : Pandas Dataframe
            Dataframe of explanatory variables
        y_value : Pandas Dataframe
            Outcome variable

    Returns
    -------
        ar_preidiction: Statsmodels model fit
            Full fit of the specified model

    """

    x_values = sm.add_constant(x_values)
    ols_model = sm.OLS(y_value, x_values)
    ols_results = ols_model.fit()

    return ols_results


def get_autoregression_prediction(ols_results, x_forecast_values):
    """
    Produces OLS estimates

    Parameters
    ----------
        ols_results : Statsmodels model fit
            Full fit of the specified model
        x_forecast_values : Pandas Dataframe
            Dataframe of lagged values to estimate next period

    Returns
    -------
        outcome_prediction[0]: int
            Prediction result

    """
    x_forecast_values = sm.add_constant(x_forecast_values, has_constant="add")
    outcome_prediction = ols_results.predict(x_forecast_values)
    return outcome_prediction[0]


def apply_autoregression(df, model_params):
    """
    Produces AR-OLS estimates for the forecasted period

    Parameters
    ----------
        df : Pandasframe
            Dataframe of available data at given time horizon
        model_params : Pandas Dataframe
            Config file of configuation settings for model

    Returns
    -------
        ar_prediction: Pandas Dataframe
            Dataframe containing information/results from the forecast

    """
    nowcast_hori = []

    forecast_period, start_estimation, end_estimation = get_variable_periods(
        df, model_params
    )  # get dates
    estimate_data, outcome_data = split_data(
        df, end_estimation, model_params
    )  # split dataframes

    # set lags
    try:
        lags_x = [model_params["p"]] * int(len(estimate_data.columns))
        lags_y = [model_params["p"]] * int(len(outcome_data.columns))
    except KeyError:
        lags_x = get_optimal_lags(estimate_data)
        lags_y = get_optimal_lags(outcome_data)

    lagged_estimate_data = apply_lags(estimate_data, lags_x)  # lag estimate data
    lagged_estimate_data_quarters = get_low_freq_rows(
        lagged_estimate_data
    )  # get quarterly rows
    lagged_outcome_data = get_lagged_outcome(
        outcome_data, lags_y[0]
    )  # get quarter lags
    all_data_ordered = combine_data(
        lagged_estimate_data_quarters, lagged_outcome_data
    )  # combine estimate data with shifted outcome data to produce estimate

    # split the data into
    if model_params["ar_type"] == "ardl":
        (
            x_estimate_values,
            y_estimate_values,
            x_forecast_values,
        ) = split_estimate_forecast(
            all_data_ordered, forecast_period, model_params["target"]
        )
    else:
        (
            x_estimate_values,
            y_estimate_values,
            x_forecast_values,
        ) = split_estimate_forecast(
            lagged_outcome_data, forecast_period, model_params["target"]
        )

    ols_results = get_autoregression_estimate(x_estimate_values, y_estimate_values)
    outcome_prediction = get_autoregression_prediction(ols_results, x_forecast_values)

    ar_results = pd.DataFrame(
        {
            "time": model_params["t"],
            "horizon": model_params["horizon"],
            "realisation": model_params["real"],
            "prediction": outcome_prediction,
            "subtype": "",
        },
        index=[0],
    )

    nowcast_hori.append(ar_results)
    ar_prediction = pd.concat(nowcast_hori, axis=1)

    return ar_prediction
