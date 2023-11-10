from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(true_values, predicted_values):
    """
    Evaluate the model performance.

    Parameters:
        true_values (pd.Series): Actual values.
        predicted_values (pd.Series): Predicted values.

    Returns:
        tuple: The Mean Squared Error and R-squared value.
    """
    mse = mean_squared_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return mse, r2
