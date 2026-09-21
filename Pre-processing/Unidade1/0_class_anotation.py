from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()

print(df1.info())

print("_________________________________________________________________")

print(hiking.head(10))
#head - mostra as 5 primeiras linhas iniciais, posso alterar no ()
print("_________________________________________________________________")

print(hiking.info())

print("_________________________________________________________________")

print(wine.info())

print("_________________________________________________________________")

print(wine.describe())
#vai mostrar dados estatísticos
print("_________________________________________________________________")

print(df1)

print("_________________________________________________________________")

print(df1.dropna())
#dropna - elimina todas as linhas com NaN

print("_________________________________________________________________")

print(df1.drop([1, 4]))
#tiro linhas específicas

print("_________________________________________________________________")

print(df1.drop("A", axis=1))
#Remover coluna específica

print("_________________________________________________________________")

print(df1.isna().sum())
#Retorna quantos elementos nulos existem no eixo

print("_________________________________________________________________")

print(df1.dropna(subset=["B"]))
#drop dos elementos nulos mas em uma coluna específica

print("_________________________________________________________________")

print(df1.dropna(thresh=2))
#drop com uma qtd de elementos nulos

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)