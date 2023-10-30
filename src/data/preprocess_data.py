import logging
import os
from pathlib import Path

from data_cleaning import clean_gdp
from dotenv import find_dotenv, load_dotenv


def preprocess():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    logging.info("Loading environment variables...")
    dotenv_path = find_dotenv()
    if not dotenv_path:
        logging.error("Could not find .env file.")
        return
    load_dotenv(dotenv_path)

    # Get project root directory
    project_root = Path(os.environ.get("PROJECT_ROOT", ""))
    if not project_root.exists():
        logging.error(f"Project root path {project_root} does not exist.")
        return

    # Define data paths
    logging.info("Defining data paths...")
    raw_data_path = project_root / "data" / "raw"
    processed_data_path = project_root / "data" / "processed"

    # Validate dataset paths
    gdp_path = raw_data_path / "GDP_CVM_SA.xls"
    # passenger_journeys_path = raw_data_path / "passenger_journeys_by_ticket_type.ods"
    # if not (gdp_path.exists() and passenger_journeys_path.exists()):
    #     logging.error("One or more dataset paths are invalid.")
    #     return

    # Data cleaning
    logging.info("Cleaning GDP data...")
    cleaned_gdp = clean_gdp(gdp_path, start_date="2000-01-01")

    # logging.info("Cleaning passenger journeys data...")
    # cleaned_passenger_journeys = clean_passenger_journeys_by_ticket_type(
    #     passenger_journeys_path
    # )

    # Save cleaned data
    logging.info("Saving cleaned data...")
    try:
        cleaned_gdp.to_csv(processed_data_path / "gdp_clean.csv", index=True)
        # cleaned_passenger_journeys.to_csv(
        #     processed_data_path / "passenger_journeys_clean.csv", index=True
        # )
        logging.info("Successfully saved cleaned data.")
    except Exception as e:
        logging.error(f"Failed to save cleaned data: {e}")


if __name__ == "__main__":
    preprocess()
