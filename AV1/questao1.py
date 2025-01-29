import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset_path = "Dataset_coletado.csv"
df = pd.read_csv(dataset_path)

print("Dados antes de tratar NaN:")
print(df.info())

df_cleaned = df.dropna()

print("Dados após remoção do NaN:")
print(df_cleaned.info())

colunas_relevantes = df_cleaned.columns[:7]
df_selected = df_cleaned[colunas_relevantes]

print("\nDataframe final com colunas relevantes:")
print(df_selected.head())

coluna_classe = df_selected.columns[-1]
print("\nDistribuição das classes:")
print(df_selected[coluna_classe].value_counts())

if df_selected[coluna_classe].dtype == 'object':
    # dtype Converter valores numéricos por códigso
    df_selected[coluna_classe] = df_selected[coluna_classe].astype('category').cat.codes
    print("\nColuna de classes convertida para valores numéricos.")

colunas_numericas = df_selected.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df_selected[colunas_numericas] = scaler.fit_transform(df_selected[colunas_numericas])
print("\nColunas numéricas normalizadas:")
print(df_selected.head())
df_selected.to_csv("Dataset_coletado.csv", index=False)
print("\nDataset atualizado 'Dataset_coletado.csv'.")

# Remover linhas com valores NaN
# Astype muda caracteras pra númericos se for solicitado