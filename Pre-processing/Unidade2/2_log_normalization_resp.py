import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

print(wine.describe())

# # Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])
#

# Print out the variance of the  total Proline column
print(np.var(wine['Proline']))

# # Check the variance of the normalized  total sulfur dioxide column
print(np.var(wine['Proline_log']))