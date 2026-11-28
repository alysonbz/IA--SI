
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Carregar os dados padronizados
dados = pd.read_csv("dataset/dados_padronizados.csv")

print("Dados carregados:")
print(dados.head())

print("\nQuantidade de amostras:", len(dados))

# 2. Testar valores de K entre 2 e 10
resultados = []

for k in range(2, 11):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    grupos = modelo.fit_predict(dados)

    inercia = modelo.inertia_
    silhouette = silhouette_score(dados, grupos)

    resultados.append({
        "K": k,
        "Inercia": inercia,
        "Silhouette": silhouette
    })

    print(
        f"K={k} | "
        f"Inércia={inercia:.2f} | "
        f"Silhouette={silhouette:.4f}"
    )

# 3. Criar tabela com os resultados
tabela = pd.DataFrame(resultados)

print("\nCOMPARAÇÃO DOS RESULTADOS:")
print(tabela.round(4))

# 4. Salvar tabela
tabela.to_csv(
    "dataset/resultados_k.csv",
    index=False
)

# 5. Gráfico do método do cotovelo
plt.figure(figsize=(8, 5))

plt.plot(
    tabela["K"],
    tabela["Inercia"],
    marker="o"
)

plt.title("Método do Cotovelo")
plt.xlabel("Quantidade de clusters (K)")
plt.ylabel("Inércia")
plt.xticks(range(2, 11))
plt.grid()

plt.tight_layout()
plt.savefig("dataset/metodo_cotovelo.png")
plt.show()

# 6. Gráfico do Silhouette Score
plt.figure(figsize=(8, 5))

plt.plot(
    tabela["K"],
    tabela["Silhouette"],
    marker="o"
)

plt.title("Silhouette Score")
plt.xlabel("Quantidade de clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))
plt.grid()

plt.tight_layout()
plt.savefig("dataset/silhouette.png")
plt.show()

print("\nTESTES DO K-MEANS CONCLUÍDOS!")


# 7. Comparar os agrupamentos com K=4 e K=10

dados_originais = pd.read_csv(
    "dataset/dados_antes_escala.csv"
)

for k in [4, 10]:

    print(f"\nAGRUPAMENTO FINAL COM K={k}")

    modelo_final = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    grupos = modelo_final.fit_predict(dados)

    # Adicionar os grupos aos dados originais
    resultado = dados_originais.copy()
    resultado["cluster"] = grupos

    # Calcular quantidade e percentual por grupo
    quantidades = resultado["cluster"].value_counts().sort_index()

    percentuais = (
        resultado["cluster"].value_counts(normalize=True).sort_index() * 100
    )

    resumo = pd.DataFrame({
        "Quantidade": quantidades,
        "Percentual": percentuais.round(2)
    })

    print(resumo)

    # Salvar os dados agrupados
    resultado.to_csv(
        f"dataset/agrupamentos_k{k}.csv",
        index=False
    )

    # Salvar o resumo dos grupos
    resumo.to_csv(
        f"dataset/resumo_clusters_k{k}.csv"
    )

print("\nCOMPARAÇÃO DOS AGRUPAMENTOS CONCLUÍDA!")
