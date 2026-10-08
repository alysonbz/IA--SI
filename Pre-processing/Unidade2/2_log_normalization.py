import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)
#mostrar dataframe
#print as caractéristicas estatísticas do dataset wine
print(wine.describe())
# estatisticas do dataset
## Aplique a função de nomarlização logarítmica na coluna Proline
proline_log=np.log(wine["Proline"])

# print a variância da coluna proline normalizada
print("var proline log")
print(np.var(proline_log))
print("var proline")
print(np.var(wine['Proline']))