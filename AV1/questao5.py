import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

file_path = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\Ferrari (20.04.23 - 01.05.24).csv"
df = pd.read_csv(file_path)
print("Dataset carregado com sucesso!")

print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais do dataset:")
print(df.info())
print("\nResumo estatístico do dataset:")
print(df.describe())

if 'Date' in df.columns:
    df = df.drop(columns=['Date'])
else:
    print("A coluna 'Date' não foi encontrada no dataset e não será excluída.")

df = df.dropna()
print("\nValores ausentes removidos.")



target = "Close"
X = df.drop(columns=[target])
y = df[target]
print(f"\nColuna alvo definida: {target}")

corr_with_target = df.corr()[target].drop(target).sort_values(ascending=False)
print("\nCorrelação com a coluna alvo:")
print(corr_with_target)

relevant_features = corr_with_target[abs(corr_with_target) > 0.1].index
print("\nAtributos relevantes selecionados (correlação > 0.1):")
print(list(relevant_features))

df = df[[*relevant_features, target]]

scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()
X_normalized = scaler_minmax.fit_transform(df.drop(columns=[target]))
X_standardized = scaler_standard.fit_transform(df.drop(columns=[target]))
print("\nDados normalizados e padronizados com sucesso.")


output_path = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\Ferrari (20.04.23 - 01.05.24).csv"
df.to_csv(output_path, index=False)
print(f"\nDataset limpo salvo em: {output_path}")


print("\nGerando gráficos de correlação e distribuições...")

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap de Correlação")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(y, kde=True)
plt.title(f"Distribuição da variável alvo '{target}'")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(X_normalized, columns=relevant_features))
plt.title("Distribuição dos Dados Normalizados")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(X_standardized, columns=relevant_features))
plt.title("Distribuição dos Dados Padronizados")
plt.show()