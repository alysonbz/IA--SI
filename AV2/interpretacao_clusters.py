
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar os dados agrupados com K=4
df = pd.read_csv("dataset(resultadps gerados)/agrupamentos_k4.csv")

# 2. Definir as características que serão analisadas
caracteristicas = [
    "temp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt"
]

# 3. Calcular as médias por cluster
medias = df.groupby("cluster")[caracteristicas].mean()

print("MÉDIAS POR CLUSTER:")
print(medias.round(2))

# 4. Calcular as medianas por cluster
medianas = df.groupby("cluster")[caracteristicas].median()

print("\nMEDIANAS POR CLUSTER:")
print(medianas.round(2))

# 5. Criar tabela comparativa
comparacao = pd.concat(
    {"Médias": medias, "Medianas": medianas},
    axis=1
)

print("\nTABELA COMPARATIVA:")
print(comparacao.round(2))

# 6. Salvar a tabela
comparacao.to_csv(
    "dataset/comparacao_clusters_k4.csv"
)

# 7. Identificar a estação e o mês mais frequentes
categorias = df.groupby("cluster").agg({
    "season": lambda x: x.mode().iloc[0],
    "mnth": lambda x: x.mode().iloc[0],
    "hr": lambda x: x.mode().iloc[0]
})

print("\nCATEGORIAS MAIS FREQUENTES:")
print(categorias)

categorias.to_csv(
    "dataset/categorias_clusters_k4.csv"
)

# 8. Gráfico das características climáticas
medias[["temp", "hum", "windspeed"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Características climáticas por cluster")
plt.xlabel("Cluster")
plt.ylabel("Média")
plt.xticks(rotation=0)
plt.legend(title="Características")
plt.tight_layout()

plt.savefig("dataset/clima_clusters_k4.png")
plt.show()
plt.close()

# 9. Gráfico da quantidade de bicicletas alugadas
medias[["casual", "registered", "cnt"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Quantidade média de aluguéis por cluster")
plt.xlabel("Cluster")
plt.ylabel("Quantidade média")
plt.xticks(rotation=0)
plt.legend(title="Tipo de usuário")
plt.tight_layout()

plt.savefig("dataset/alugueis_clusters_k4.png")
plt.show()
plt.close()

