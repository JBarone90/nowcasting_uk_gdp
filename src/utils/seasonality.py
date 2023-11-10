import statsmodels.api as sm
from scipy.stats import chi2


def qs_test(data, seasonal_lag, k=2):
    """
    Perform the QS (Seasonality) Test on autocovariance at seasonal lags.
    The QS test is a variant of the Ljung-Box test computed on seasonal lags, where we only consider positive auto-correlations.

    This function calculates the QS (Seasonality) statistic and its associated p-value to test for seasonality in a time series at a specified seasonal lag.

    Parameters:
    - data: numpy array or pandas Series
        A time series dataset for which you want to test seasonality.

    - seasonal_lag: int
        The seasonal lag at which to check for autocovariance. For example, if analyzing monthly data, a seasonal lag of 12 would be appropriate.

    - k: int, optional (default=2)
        The number of terms in the summation of the QS statistic. A higher 'k' may capture more complex seasonality patterns.

    Returns:
    - qs_statistic: float
        The computed QS statistic, which measures the strength of seasonality in the data.

    - p_value: float
        The p-value associated with the test. A lower p-value suggests stronger evidence against the null hypothesis of no seasonality.

    Notes:
    - The QS statistic is calculated based on the autocovariances of the data at multiples of the specified seasonal lag.

    - The p-value is calculated using a chi-squared distribution with degrees of freedom equal to 'k'.

    Example:
    ```
    import numpy as np
    from your_module import qs_test

    # Load your time series data (e.g., monthly data)
    data = np.array([...])

    # Perform the QS Test with a seasonal lag of 12 (for monthly data) and k=3
    qs_statistic, p_value = qs_test(data, seasonal_lag=12, k=3)

    if p_value < 0.05:
        print("Seasonality detected.")
    else:
        print("No significant seasonality."
    ```

    References:
    - [https://jdemetra-new-documentation.netlify.app/m-tests#seasonality-tests]

    """
    n = len(data)

    # Calculate the QS statistic using the equation
    qs_statistic = 0
    for i in range(1, k + 1):
        # Calculate autocovariance at i*l
        autocovariances = sm.tsa.acovf(data, nlag=seasonal_lag * i)
        gamma_i_l = autocovariances[-1]
        # Add the contribution of the current term to the QS statistic
        qs_statistic += (max(0, gamma_i_l) ** 2) / (n - i * seasonal_lag)

    qs_statistic *= n * (n + 2)

    # Calculate the p-value using a chi-squared distribution with 'k' degrees of freedom
    p_value = 1.0 - chi2.cdf(qs_statistic, df=k)

    return qs_statistic, p_value
