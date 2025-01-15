import pandas as pd
import numpy as np

# 1. Carregar o dataset
try:
    df = pd.read_csv('/home/kali/Downloads/gender_classification_v7.xls')
    print("Dataset carregado.")
except FileNotFoundError:
    print("Arquivo não encontrado.")

print("Verificando células vazias...")
if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print("Linhas com NaN foram excluídas.")
else:
    print("Não há células vazias.")
print("\nColunas do dataset:")
print(df.columns)
columns_needed = ['long_hair', 'forehead_width_cm', 'gender']
df = df[columns_needed]
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
print("\nDataframe final:")
print(df.head())
print("\nDistribuição de classes:")
print(df['gender'].value_counts(normalize=True))
# 7. Salvar o dataset atualizado
df.to_csv('gender_classification_ajustado.csv', index=False)
print("Dataset atualizado salvo como 'gender_classification_ajustado.csv'")
