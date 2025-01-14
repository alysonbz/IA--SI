import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\star_classification.csv")

print("Valores faltantes por coluna:")
print(df.isnull().sum())

df_cleaned = df.dropna()

print("Valores faltantes após limpeza:")
print(df_cleaned.isnull().sum())

colunas_relevantes = ['obj_ID', 'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
                      'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID',
                      'class', 'redshift', 'plate', 'MJD', 'fiber_ID']

colunas_existentes = [col for col in colunas_relevantes if col in df_cleaned.columns]
df_final = df_cleaned[colunas_existentes]

print("DataFrame final:")
print(df_final.head())

print("Distribuição das classes:")
print(df_final['class'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df_final)
plt.title('Distribuição das Classes')
plt.xlabel('Classe')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)
plt.show()

if df_final['class'].dtype == 'object':
    # Converter as classes para valores numéricos
    df_final['class'] = df_final['class'].astype('category').cat.codes
    print("Coluna 'class' após conversão para numérico:")
    print(df_final['class'].head())

df_final.to_csv(r"C:\Users\bende\SI\IA\IA--SI\IA--SI\star_classification_ajustado.csv", index=False)

print("Dataset salvo como 'star_classification_ajustado.csv'.")