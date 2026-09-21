import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

#print as caractéristicas estatísticas do dataset wine
print(wine.describe())

print("____________________________________________________________")

#normalização - aplicar o mesmo peso para as colunas
## Aplique a função de nomarlização logarítmica na coluna Proline
wine['Proline'] = np.log(wine['Proline'])

print("____________________________________________________________")

# Print a variância da coluna proline
print(wine['Proline'])

print("____________________________________________________________")

# print a variância da coluna proline normalizada
print(np.var(wine['Proline']))