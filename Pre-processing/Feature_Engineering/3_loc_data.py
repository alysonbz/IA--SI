from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()
# Use .loc to create a mean column
wine["mean"] = wine.loc[:,'Alcohol':].mean(axis=1)

# Take a look at the results
pd.set_option('display.max_columns', None)
print(wine.head())