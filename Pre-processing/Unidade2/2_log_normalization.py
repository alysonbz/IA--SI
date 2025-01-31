import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

# Print as características estatísticas do dataset wine
print(wine.describe())

# Aplique a função de normalização logarítmica na coluna Proline
wine['Proline_log'] = np.log(wine['Proline'])

# Print a variância da coluna Proline
print(np.var(wine['Proline']))

# Print a variância da coluna Proline normalizada
print(np.var(wine['Proline_log']))
