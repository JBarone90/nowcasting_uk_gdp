import pandas as pd


def clean_gdp(
    file_path,
    sheet_name="data",
    start_date="2003-10-01",
):
    """
    Reads a GDP Excel Spreadsheet, cleans, and formats the data into a time series DataFrame.

    Parameters:
    file_path (str): Path to the Excel file containing GDP data.
    sheet_name (str): Name of the sheet containing the data (default is "data").
    start_date (str): Starting date for filtering the data (default is "2003-10-01").

    Returns:
    pandas.DataFrame: A cleaned and formatted time series DataFrame with date as index and GDP values.
    """

    # Load data from Excel file
    gdp_df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=range(7),
        usecols=[0, 1],
    )
    gdp_df.columns = ["date", "value"]

    # Filter rows with quarterly data (assuming date format contains 'Q')
    gdp_df = gdp_df[gdp_df["date"].str.contains("Q", na=False)]

    # Replace spaces with hyphens and convert to datetime, setting errors='coerce' to handle misformatted dates
    gdp_df["date"] = pd.to_datetime(
        gdp_df["date"].str.replace(" ", "-"), errors="coerce"
    )

    # Drop any rows with NaT (Not a Timestamp) due to misformatted dates
    gdp_df = gdp_df.dropna(subset=["date"])

    # Set date as the index
    gdp_df.set_index("date", inplace=True)

    # Filter data from the specified start date
    gdp_df = gdp_df.loc[start_date:]

    return gdp_df
