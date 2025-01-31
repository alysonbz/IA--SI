import pandas as pd
import numpy as np

# 1) Importar bibliotecas e carregar o dataset
# Criando o dataset a partir do exemplo fornecido
data = pd.read_csv('waterquality.csv')
# Criar DataFrame
df = pd.DataFrame(data)

# 2) Verificar células vazias, zeros, NaN ou strings
if df.isnull().any().any() or (df == 0).any().any() or df.applymap(lambda x: isinstance(x, str)).any().any():
    df = df[(df != 0).all(axis=1)].dropna()

# 3) Selecionar colunas relevantes (ammonia, chloramine, lead, nitrates, is_safe)
selected_columns = ['ammonia', 'chloramine', 'lead', 'nitrates', 'is_safe']
df_selected = df[selected_columns]

# 4) Exibir o DataFrame final e a distribuição de classes
print("DataFrame Final:\n", df_selected)
print("\nDistribuição de Classes:\n", df_selected['is_safe'].value_counts())

# 5) Avaliar necessidade de mais pré-processamento
# Exemplo: Normalização para as colunas contínuas (se necessário)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_selected[['ammonia', 'chloramine', 'lead', 'nitrates']] = scaler.fit_transform(df_selected[['ammonia', 'chloramine', 'lead', 'nitrates']])
# 6) Salvar o dataset ajustado em um novo arquivo CSV
df_selected.to_csv('waterquality_ajustado.csv', index=False)