import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer["start_date_date"])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].dt.month

# Take a look at the converted and new month columns
print(volunteer[['start_date_date','start_date_converted', 'start_date_month']].head())