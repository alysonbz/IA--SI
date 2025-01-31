import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Carregar o dataset
file_path = 'car_price.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Arquivo {file_path} não encontrado.")
    exit()

# 2) Selecionar as 5 colunas mais importantes
selected_columns = ['Price', 'Levy', 'Prod. year', 'Airbags', 'Leather interior']
df = df[selected_columns]

# 3) Substituir 'Yes' por 1 e 'No' por 0
if 'Leather interior' in df.columns:
    df['Leather interior'] = (df['Leather interior'].replace({'Yes': 1, 'No': 0}))

# 4) Remover linhas com valores não numéricos
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# 5) Verificar correlação entre atributos e o alvo
target = 'Price'
correlation = df.corr()
target_correlation = correlation[target].sort_values(ascending=False)
print("Correlação dos atributos com o alvo (Price):\n", target_correlation)

# 6) Visualizar a matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=True)
plt.title('Matriz de Correlação')
plt.show()

# 7) Plotar relação entre os atributos e o alvo
for feature in df.columns:
    if feature != target:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[feature], y=df[target])
        plt.title(f"Relação entre {feature} e {target}")
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

# 8) Dataset ajustado
print("Dataset ajustado:\n", df.head())