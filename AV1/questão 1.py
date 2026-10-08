import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

#1. Carregando o dataset HCV
hcv_data = fetch_ucirepo(id=571)
X = hcv_data.data.features
y = hcv_data.data.targets

#Juntando as features e o target
df = pd.concat([X, y], axis=1)

#Item 1: Identificação de dimensões, tipos e variável classe
print("Dimensões originais do dataset (Linhas e Colunas):")
print(df.shape)
print("\nTipos de dados de cada coluna:")
print(df.dtypes)
target_col = y.columns[0]
print("\nVariável utilizada como classe:")
print(target_col)

#Item 2: Verificação de inconsistências (valores nulos)
print("\nVerificação de inconsistências (Valores Nulos):")
print(df.isnull().sum())

#Tratamento: Remoção dos valores nulos para garantir a precisão do cálculo de distâncias do KNN
df_clean = df.dropna().copy()
print("\nDimensões após a remoção dos nulos:")
print(df_clean.shape)

#Item 5: Avaliação das escalas e transformações necessárias
print("\nAvaliação das escalas e transformações necessárias (Mínimo e Máximo):")
print(df_clean.select_dtypes(include=[np.number]).agg(['min', 'max']).T)

#Guardando o dataset
df_clean.to_csv('hcv_dataset_preparado.csv', index=False)
