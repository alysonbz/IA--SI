# ==============================================================================
# AV2 – INTELIGÊNCIA ARTIFICIAL
# Alunos: João Victor e Antônio Erick
# Dataset: Student Performance (UCI, id=320) | Alvo: G3
# ==============================================================================

# Importa bibliotecas básicas para cálculo numérico, manipulação de tabelas e gráficos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importa função para baixar a base de dados diretamente do repositório da UCI
from ucimlrepo import fetch_ucirepo

# Importa ferramentas nativas do scikit-learn para pré-processamento e agrupamento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configura o pandas para exibir todas as colunas e usar largura maior ao imprimir tabelas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# ==============================================================================
# 0. CARREGAMENTO DOS DADOS
# ==============================================================================

# Baixa o conjunto de dados "Student Performance" (ID 320) da UCI
base_uci_alunos = fetch_ucirepo(id=320)

# Junta as colunas de características (features) com as colunas de notas (targets) lado a lado
tabela_alunos = pd.concat(
    [base_uci_alunos.data.features, base_uci_alunos.data.targets],
    axis=1
)


# ==============================================================================
# MOMENTO I – APRENDIZADO NÃO SUPERVISIONADO
# ==============================================================================

# ------------------------------------------------------------------------------
# 3. ANÁLISE EXPLORATÓRIA DOS DADOS
# ------------------------------------------------------------------------------
print("\n--- 3. ANÁLISE EXPLORATÓRIA ---")

# 3a) Exibe quantidade de linhas e colunas da tabela e informações de tipos de dados
print("Dimensões (amostras, colunas):", tabela_alunos.shape)
tabela_alunos.info()

# Separa os nomes das colunas numéricas e das colunas categóricas (texto) em listas
colunas_numericas = tabela_alunos.select_dtypes(include=[np.number]).columns.tolist()
colunas_categoricas = tabela_alunos.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Numéricas ({len(colunas_numericas)}):", colunas_numericas)
print(f"Categóricas ({len(colunas_categoricas)}):", colunas_categoricas)

# 3b) Soma todos os valores ausentes (NaN) na tabela e conta quantas linhas são duplicadas
print("\nValores ausentes (total):", tabela_alunos.isna().sum().sum())
print("Linhas duplicadas:", tabela_alunos.duplicated().sum())
# Obs.: todas as notas estão entre 0 e 20, mas existem alunos com G3 == 0.

# 3c) Calcula estatísticas descritivas básicas e adiciona a mediana como coluna extra
resumo_estatistico = tabela_alunos[colunas_numericas].describe().T
resumo_estatistico["mediana"] = tabela_alunos[colunas_numericas].median()
print("\nEstatísticas descritivas:\n", resumo_estatistico.round(2))

# 3d) Seleciona variáveis principais e gera histogramas para ver a distribuição dos valores
colunas_principais = ["age", "studytime", "failures", "absences", "G3"]
tabela_alunos[colunas_principais].hist(bins=15, figsize=(12, 7))
plt.suptitle("3d) Histogramas – Principais Variáveis")
plt.tight_layout()
plt.show()

# Cria uma figura com 1 linha e 5 gráficos lado a lado para os boxplots
figura_boxplots, eixos_boxplots = plt.subplots(1, len(colunas_principais), figsize=(14, 4))
# Percorre cada eixo e coluna desenhando o boxplot correspondente
for eixo_atual, nome_coluna in zip(eixos_boxplots, colunas_principais):
    sns.boxplot(y=tabela_alunos[nome_coluna], ax=eixo_atual)
    eixo_atual.set_title(nome_coluna)
plt.suptitle("3d) Boxplots – Principais Variáveis")
plt.tight_layout()
plt.show()

# 3e) Cria 4 gráficos de dispersão comparando variáveis escolhidas contra a nota final G3
colunas_dispersao = ["studytime", "failures", "absences", "age"]
figura_dispersao, eixos_dispersao = plt.subplots(1, len(colunas_dispersao), figsize=(16, 4))
for eixo_atual, nome_coluna in zip(eixos_dispersao, colunas_dispersao):
    sns.scatterplot(x=tabela_alunos[nome_coluna], y=tabela_alunos["G3"], alpha=0.5, ax=eixo_atual)
    eixo_atual.set_title(f"{nome_coluna} × G3")
plt.tight_layout()
plt.show()

# Calcula e plota o mapa de calor (heatmap) da matriz de correlação das variáveis numéricas
plt.figure(figsize=(12, 9))
matriz_correlacao_geral = tabela_alunos[colunas_numericas].corr()
sns.heatmap(
    matriz_correlacao_geral,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    annot_kws={"size": 7}
)
plt.title("3e) Matriz de Correlação")
plt.tight_layout()
plt.show()

# 3f) Detecta outliers (valores extremos) usando a regra do intervalo interquartil (IQR)
# Outliers deslocam os centróides do K-Means e distorcem a reta da Regressão Linear
primeiro_quartil = tabela_alunos[colunas_numericas].quantile(0.25)
terceiro_quartil = tabela_alunos[colunas_numericas].quantile(0.75)
intervalo_interquartil = terceiro_quartil - primeiro_quartil

limite_inferior = primeiro_quartil - 1.5 * intervalo_interquartil
limite_superior = terceiro_quartil + 1.5 * intervalo_interquartil

# Conta quantos valores ficam abaixo do limite inferior ou acima do limite superior em cada coluna
contagem_outliers = (
    (tabela_alunos[colunas_numericas] < limite_inferior) |
    (tabela_alunos[colunas_numericas] > limite_superior)
).sum()

# Monta uma tabela com a contagem absoluta e o percentual de outliers por variável
tabela_outliers = pd.DataFrame({
    "outliers": contagem_outliers,
    "percentual (%)": (contagem_outliers / len(tabela_alunos) * 100).round(2)
})
print("\n3f) Outliers por variável (IQR):\n", tabela_outliers)

# 3g) 3 achados principais dessa etapa (para comentar na apresentação):
# 1. Quem já reprovou antes ('failures') tende a tirar notas finais bem menores.
# 2. A maioria falta pouco, mas existe uma minoria com muitas faltas ('absences' tem vários outliers).
# 3. A base não tem valores ausentes nem linhas repetidas, mas há alunos com nota zero no fim do ano.


# ------------------------------------------------------------------------------
# 4. PRÉ-PROCESSAMENTO DOS DADOS (PARA K-MEANS)
# ------------------------------------------------------------------------------

# 4b) Cria uma cópia da tabela garantindo que não há registros duplicados
tabela_alunos_limpa = tabela_alunos.drop_duplicates().copy()

# 4a) Remove G1, G2 e G3 diretamente para agrupar os alunos apenas por perfil, hábitos e rotina
dados_entrada_brutos = tabela_alunos_limpa.drop(["G1", "G2", "G3"], axis=1)

# Salva uma cópia na escala original (idade real, faltas reais) para interpretar os grupos depois
tabela_clusters_original = dados_entrada_brutos.copy()

# Identifica quais colunas de entrada são numéricas e quais são categóricas
colunas_numericas_entrada = dados_entrada_brutos.select_dtypes(include=[np.number]).columns.tolist()
colunas_categoricas_entrada = dados_entrada_brutos.select_dtypes(exclude=[np.number]).columns.tolist()

# 4c) Usa ColumnTransformer + OneHotEncoder (nativos do scikit-learn) para transformar textos
# em colunas binárias (0 e 1), descartando a primeira categoria para evitar multicolinearidade
codificador_colunas = ColumnTransformer(
    transformers=[
        ("numericas", "passthrough", colunas_numericas_entrada),
        (
            "categoricas",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            colunas_categoricas_entrada,
        ),
    ],
    verbose_feature_names_out=False,
)

# 4d e 4e) Encadeia codificação + padronização (média 0, desvio-padrão 1) em um Pipeline nativo
# A padronização evita que variáveis com escala grande (ex.: faltas) dominem as distâncias do K-Means
pipeline_preprocessamento_cluster = Pipeline(
    steps=[
        ("codificacao", codificador_colunas),
        ("padronizacao", StandardScaler()),
    ]
)

# Ajusta e transforma os dados de entrada em matriz numérica padronizada para o K-Means
dados_cluster_padronizados = pipeline_preprocessamento_cluster.fit_transform(dados_entrada_brutos)


# ------------------------------------------------------------------------------
# 5. APLICAÇÃO DO K-MEANS
# ------------------------------------------------------------------------------
print("\n--- 5. K-MEANS ---")

# 5a) Define a faixa de valores de K (de 2 até 10 grupos) e listas para guardar as métricas
valores_k = list(range(2, 11))
lista_inercias = []
lista_silhuetas = []

# Testa cada valor de K treinando o K-Means e calculando inércia e coeficiente de silhueta
for quantidade_grupos in valores_k:
    modelo_kmeans_teste = KMeans(
        n_clusters=quantidade_grupos,
        random_state=42,
        n_init=10
    )
    rotulos_previstos = modelo_kmeans_teste.fit_predict(dados_cluster_padronizados)
    # 5b) Inércia mede a soma das distâncias quadradas internas (coesão dos grupos)
    lista_inercias.append(modelo_kmeans_teste.inertia_)
    # 5d) Silhouette Score mede o quão bem separados os grupos estão entre si
    lista_silhuetas.append(silhouette_score(dados_cluster_padronizados, rotulos_previstos))

# Organiza os resultados de K, Inércia e Silhouette em um DataFrame
tabela_avaliacao_k = pd.DataFrame({
    "K": valores_k,
    "Inercia": np.round(lista_inercias, 2),
    "Silhouette": np.round(lista_silhuetas, 4)
})
print(tabela_avaliacao_k)

# 5c) Plota lado a lado o gráfico do Método do Cotovelo (Inércia) e o da Silhueta
figura_metricas_k, eixos_metricas_k = plt.subplots(1, 2, figsize=(14, 4.5))

eixos_metricas_k[0].plot(valores_k, lista_inercias, marker="o")
eixos_metricas_k[0].set_title("5c) Método do Cotovelo (Inércia × K)")
eixos_metricas_k[0].set_xlabel("K")
eixos_metricas_k[0].set_ylabel("Inércia")

eixos_metricas_k[1].plot(valores_k, lista_silhuetas, marker="s", color="green")
eixos_metricas_k[1].set_title("5d) Silhouette Score × K")
eixos_metricas_k[1].set_xlabel("K")
eixos_metricas_k[1].set_ylabel("Silhouette")

plt.tight_layout()
plt.show()

# 5e) Seleciona automaticamente o valor de K que obteve o maior Silhouette Score
indice_melhor_silhueta = int(np.argmax(lista_silhuetas))
melhor_k = valores_k[indice_melhor_silhueta]
print(f"\n5e) K escolhido (maior Silhouette): {melhor_k}")

# 5f) Treina o modelo K-Means definitivo com o melhor K e atribui cada aluno a um cluster
modelo_kmeans_final = KMeans(
    n_clusters=melhor_k,
    random_state=42,
    n_init=10
)
clusters_finais = modelo_kmeans_final.fit_predict(dados_cluster_padronizados)

# Converte os rótulos finais em uma Series alinhada ao índice da tabela limpa
serie_clusters = pd.Series(clusters_finais, index=tabela_alunos_limpa.index, name="cluster")

# 5g) Calcula a quantidade absoluta e o percentual de alunos em cada cluster
tabela_distribuicao_clusters = pd.DataFrame({
    "Quantidade": serie_clusters.value_counts().sort_index(),
    "Percentual (%)": (serie_clusters.value_counts(normalize=True).sort_index() * 100).round(2)
})
print("\n5g) Distribuição dos clusters:\n", tabela_distribuicao_clusters)


# ------------------------------------------------------------------------------
# 6. INTERPRETAÇÃO DOS CLUSTERS
# ------------------------------------------------------------------------------
print("\n--- 6. INTERPRETAÇÃO DOS CLUSTERS ---")

# Adiciona a coluna com o número do cluster na tabela de valores originais
tabela_clusters_original["cluster"] = clusters_finais

# Lista todas as colunas numéricas da tabela, exceto a própria coluna 'cluster'
colunas_numericas_cluster = [
    coluna for coluna in tabela_clusters_original.select_dtypes(include=np.number).columns
    if coluna != "cluster"
]

# 6a e 6b) Calcula média e mediana de cada variável numérica por cluster na escala original
medias_por_cluster = tabela_clusters_original.groupby("cluster")[colunas_numericas_cluster].mean().round(2).T
medianas_por_cluster = tabela_clusters_original.groupby("cluster")[colunas_numericas_cluster].median().round(2).T

print("\n6a/6b) Médias por cluster (escala original):\n", medias_por_cluster)
print("\n6a/6b) Medianas por cluster (escala original):\n", medianas_por_cluster)

# 6c) Compara visualmente faltas e tempo de estudo entre os clusters via boxplots
figura_perfis, eixos_perfis = plt.subplots(1, 2, figsize=(12, 4.5))

sns.boxplot(x="cluster", y="absences", data=tabela_clusters_original, ax=eixos_perfis[0])
eixos_perfis[0].set_title("6c) Faltas (absences) por cluster")

sns.boxplot(x="cluster", y="studytime", data=tabela_clusters_original, ax=eixos_perfis[1])
eixos_perfis[1].set_title("6c) Tempo de estudo (studytime) por cluster")

plt.tight_layout()
plt.show()

# 6d) Principais diferenças entre grupos: consumo de álcool, tempo de estudo, escolaridade dos pais e faltas.
# 6e) Na apresentação, nomear o perfil de cada cluster com base nas médias e medianas acima.
# 6f) Os grupos fazem sentido na prática, mas muitas colunas binárias (0/1) aumentam a
#     dimensionalidade e naturalmente reduzem o valor do Silhouette Score.


# ==============================================================================
# MOMENTO II – APRENDIZADO SUPERVISIONADO: REGRESSÃO
# ==============================================================================

# ------------------------------------------------------------------------------
# 7. DEFINIÇÃO DO PROBLEMA DE REGRESSÃO
# ------------------------------------------------------------------------------
print("\n--- 7. DEFINIÇÃO DO PROBLEMA ---")

# 7a, 7b e 7e) Separa a variável alvo (G3) das variáveis preditoras (removendo G1, G2 e G3)
alvo_regressao = tabela_alunos_limpa["G3"]
preditoras_regressao = tabela_alunos_limpa.drop(["G1", "G2", "G3"], axis=1)

# Recupera o codificador já ajustado para gerar a tabela numérica com os nomes das colunas codificadas
codificador_ajustado = pipeline_preprocessamento_cluster.named_steps["codificacao"]
matriz_preditoras_codificadas = codificador_ajustado.transform(preditoras_regressao)
nomes_colunas_codificadas = codificador_ajustado.get_feature_names_out()

tabela_preditoras_codificadas = pd.DataFrame(
    matriz_preditoras_codificadas,
    columns=nomes_colunas_codificadas,
    index=preditoras_regressao.index
)

# 7c) Calcula a correlação de cada preditora codificada com G3 e ordena pelo valor absoluto
correlacao_com_alvo = (
    tabela_preditoras_codificadas
    .corrwith(alvo_regressao)
    .sort_values(key=np.abs, ascending=False)
)
print("\n7c) Maiores correlações com G3:\n", correlacao_com_alvo.head(10).round(3))

# Plota gráfico de barras horizontais com as 15 variáveis mais correlacionadas com G3
plt.figure(figsize=(8, 6))
correlacao_com_alvo.head(15).sort_values().plot(kind="barh")
plt.title("7c) Correlação das preditoras com G3 (15 maiores)")
plt.tight_layout()
plt.show()

# 7d) Verifica correlação entre variáveis preditoras numéricas (possível multicolinearidade)
# Quando preditoras são muito correlacionadas, a Regressão Linear fica instável e Ridge/Lasso ajudam
colunas_preditoras_numericas = preditoras_regressao.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(10, 8))
sns.heatmap(
    preditoras_regressao[colunas_preditoras_numericas].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    annot_kws={"size": 7}
)
plt.title("7d) Correlação entre variáveis independentes")
plt.tight_layout()
plt.show()