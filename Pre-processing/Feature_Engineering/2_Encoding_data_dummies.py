import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
pd.set_option('display.max_columns', None)
print(category_enc.head())