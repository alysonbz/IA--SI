
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o dataset
df = pd.read_csv("dataset/hour.csv")

# 2. Mostrar as primeiras linhas
print("PRIMEIRAS LINHAS:")
print(df.head())

# 3. Quantidade de registros e colunas
print("\nDIMENSÕES DO DATASET:")
print("Amostras:", df.shape[0])
print("Características:", df.shape[1])

# 4. Tipos de dados
print("\nTIPOS DE DADOS:")
print(df.dtypes)

# 5. Informações gerais
print("\nINFORMAÇÕES GERAIS:")
df.info()

# 6. Valores ausentes
print("\nVALORES AUSENTES:")
print(df.isnull().sum())

# 7. Registros duplicados
print("\nREGISTROS DUPLICADOS:")
print(df.duplicated().sum())

# 8. Verificar inconsistências
print("\nINCONSISTÊNCIAS:")
print(
    (df["cnt"] != df["casual"] + df["registered"]).sum()
)

# 9. Estatísticas descritivas
print("\nESTATÍSTICAS DESCRITIVAS:")
print(df.describe().round(2))

# 10. Histogramas
df[["temp", "hum", "windspeed", "cnt"]].hist(
    figsize=(10, 8),
    bins=20
)

plt.tight_layout()
plt.savefig("dataset/histogramas.png")
plt.show()

# 11. Boxplots
plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df[["temp", "hum", "windspeed", "cnt"]]
)

plt.title("Distribuição e valores discrepantes")
plt.tight_layout()
plt.savefig("dataset/boxplots.png")
plt.show()

# 12. Temperatura e quantidade de bicicletas
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="temp",
    y="cnt"
)

plt.title("Temperatura x Bicicletas alugadas")
plt.xlabel("Temperatura")
plt.ylabel("Quantidade de bicicletas")
plt.tight_layout()
plt.savefig("dataset/temperatura_bicicletas.png")
plt.show()

# 13. Matriz de correlação
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matriz de correlação")
plt.tight_layout()
plt.savefig("dataset/correlacao.png")
plt.show()

print(df.describe().round(2))
variaveis = ['temp', 'hum', 'windspeed',
             'casual', 'registered', 'cnt']

print(df[variaveis].describe().round(2).to_string())