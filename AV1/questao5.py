import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Carregar o dataset
file_path = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\Ferrari (20.04.23 - 01.05.24).csv"
df = pd.read_csv(file_path)
print("Dataset carregado com sucesso!")

# Exibir as primeiras linhas e informações gerais do dataset
print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais do dataset:")
print(df.info())
print("\nResumo estatístico do dataset:")
print(df.describe())

# Remover valores ausentes e colunas irrelevantes
# Remover valores ausentes e colunas irrelevantes
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])
else:
    print("A coluna 'Date' não foi encontrada no dataset e não será excluída.")

df = df.dropna()
print("\nValores ausentes removidos.")


# Definir a coluna alvo e separar variáveis independentes e dependentes
target = "Close"
X = df.drop(columns=[target])
y = df[target]
print(f"\nColuna alvo definida: {target}")

# Calcular correlação e selecionar atributos relevantes
corr_with_target = df.corr()[target].drop(target).sort_values(ascending=False)
print("\nCorrelação com a coluna alvo:")
print(corr_with_target)

# Selecionar atributos relevantes com base no limiar de correlação
relevant_features = corr_with_target[abs(corr_with_target) > 0.1].index
print("\nAtributos relevantes selecionados (correlação > 0.1):")
print(list(relevant_features))

# Atualizar o dataframe com os atributos relevantes
df = df[[*relevant_features, target]]

# Normalizar e padronizar os dados
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()
X_normalized = scaler_minmax.fit_transform(df.drop(columns=[target]))
X_standardized = scaler_standard.fit_transform(df.drop(columns=[target]))
print("\nDados normalizados e padronizados com sucesso.")

# Salvar dataset limpo
output_path = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\Ferrari (20.04.23 - 01.05.24).csv"
df.to_csv(output_path, index=False)
print(f"\nDataset limpo salvo em: {output_path}")

# Visualizações
print("\nGerando gráficos de correlação e distribuições...")

# Heatmap de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap de Correlação")
plt.show()

# Distribuição da variável alvo
plt.figure(figsize=(8, 6))
sns.histplot(y, kde=True)
plt.title(f"Distribuição da variável alvo '{target}'")
plt.show()

# Distribuição dos dados normalizados
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(X_normalized, columns=relevant_features))
plt.title("Distribuição dos Dados Normalizados")
plt.show()

# Distribuição dos dados padronizados
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(X_standardized, columns=relevant_features))
plt.title("Distribuição dos Dados Padronizados")
plt.show()