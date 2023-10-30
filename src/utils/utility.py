import json

from sklearn.metrics import mean_squared_error, r2_score


def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


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
