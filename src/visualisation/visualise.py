import matplotlib.pyplot as plt
import pandas as pd


def plot_ar_results(train_data, ar_prediction, plot_confidence_intervals=False):
    """
    Plot the training data, forecasts, and test data for an autoregressive model.

    Parameters:
        train_data (pd.DataFrame): The training data.
        ar_prediction (pd.DataFrame): The predictions from the autoregressive model along with test data.
        plot_confidence_intervals (bool): Whether to plot confidence intervals. Default is False.

    """
    # Ensure train_data is a pandas Series
    if isinstance(train_data, pd.DataFrame) and train_data.shape[1] == 1:
        train_data = train_data.squeeze()

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the training data in green
    plt.plot(train_data.index, train_data, color="green", label="Training Data")

    # Plot the forecasts in red
    plt.plot(
        ar_prediction.index,
        ar_prediction["Prediction"],
        color="red",
        label="Predictions",
    )

    # Plot the test data in blue
    plt.plot(
        ar_prediction.index, ar_prediction["Actual"], color="blue", label="Test Data"
    )

    # Optionally plot the confidence intervals
    if plot_confidence_intervals:
        plt.fill_between(
            ar_prediction.index,
            ar_prediction["Lower CI"],
            ar_prediction["Upper CI"],
            color="pink",
            alpha=0.3,
            label="95% Confidence Interval",
        )

    # Add labels and legend
    plt.xlabel("Date")
    plt.ylabel("Quarter-on-Quarter Change")
    plt.title("Quarterly Growth Analysis using Autoregressive Model")
    plt.legend()
    plt.grid(True)
    plt.show()
