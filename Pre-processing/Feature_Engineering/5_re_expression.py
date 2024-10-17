from src.utils import load_hiking_dataset
import re

# Write a pattern to extract numbers and decimals
def return_mileage(length):
    # Search the text for matches
    try:
        mile = re.search("\d+\.\d+", length)
    except:
        mile = None
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))

hiking = load_hiking_dataset()
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(return_mileage)
print(hiking[["Length", "Length_num"]].head())