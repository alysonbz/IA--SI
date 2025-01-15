import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

#print as caractéristicas estatísticas do dataset wine
print(wine.info())

## Aplique a função de nomarlização logarítmica na coluna Proline
<<<<<<< HEAD
wine['Proline_log'] = np.log(wine['Proline'])


# Print a variância da coluna proline
print(np.var(wine['Proline_log']))
=======
wine['Profile_log'] = np.log(wine['Proline'])
#
# Print a variância da coluna proline
print(wine['Profile_log'].var())
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

# print a variância da coluna proline normalizada
print(np.var(wine['Proline_log']))