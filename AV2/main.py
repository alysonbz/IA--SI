import pandas as pd
import numpy as np
import os

os.makedirs("graficos", exist_ok=True)

# ==========================================================
# TRABALHO - AIR QUALITY UCI
# ANÁLISE EXPLORATÓRIA DOS DADOS
# ==========================================================

import pandas as pd

# Carregar o dataset
dados = pd.read_csv(
    "dados/AirQualityUCI.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)

# Remover linhas completamente vazias
dados = dados.dropna(how="all")

# Remover colunas completamente vazias
dados = dados.dropna(axis=1, how="all")

# Exibir as primeiras linhas
print("PRIMEIRAS LINHAS DO DATASET:")
print(dados.head())

# Exibir dimensões
print("\nDIMENSÕES DO DATASET:")
print(dados.shape)

# Exibir informações
print("\nINFORMAÇÕES DO DATASET:")
dados.info()

# ==========================================================
# INVESTIGAÇÃO DOS VALORES AUSENTES (-200)
# ==========================================================

# Contar quantos valores -200 existem em cada coluna numérica
valores_ausentes = (dados.select_dtypes(include="number") == -200).sum()

# Calcular a porcentagem de valores -200 em cada coluna
porcentagem_ausentes = (
    valores_ausentes / len(dados) * 100
)

# Organizar os resultados em uma tabela
analise_ausentes = pd.DataFrame({
    "Quantidade de -200": valores_ausentes,
    "Porcentagem (%)": porcentagem_ausentes
})

print("\nANÁLISE DOS VALORES AUSENTES (-200):")
print(analise_ausentes)

# ==========================================================
# SUBSTITUIR -200 POR VALORES AUSENTES (NaN)
# ==========================================================

# Identificar as colunas numéricas
colunas_numericas = dados.select_dtypes(include="number").columns

# Substituir -200 por NaN
dados[colunas_numericas] = dados[colunas_numericas].replace(-200, np.nan)

# Verificar a quantidade de valores ausentes
print("\nQUANTIDADE DE VALORES AUSENTES POR COLUNA:")
print(dados.isnull().sum())

# Verificar a porcentagem de valores ausentes
print("\nPORCENTAGEM DE VALORES AUSENTES:")
print((dados.isnull().mean() * 100).round(2))

# ==========================================================
# VERIFICAÇÃO DE REGISTROS DUPLICADOS
# ==========================================================

# Verificar linhas completamente duplicadas
duplicados = dados.duplicated().sum()

# Verificar datas e horários duplicados
duplicados_data_hora = dados.duplicated(
    subset=["Date", "Time"]
).sum()

print("\nANÁLISE DE REGISTROS DUPLICADOS:")
print("Linhas completamente duplicadas:", duplicados)
print("Registros com data e horário duplicados:", duplicados_data_hora)

# ==========================================================
# ESTATÍSTICAS DESCRITIVAS
# ==========================================================

# Selecionar apenas as variáveis numéricas
dados_numericos = dados.select_dtypes(include="number")

# Calcular estatísticas descritivas
estatisticas = dados_numericos.describe().T

# Adicionar a mediana
estatisticas["median"] = dados_numericos.median()

# Exibir as estatísticas solicitadas
print("\nESTATÍSTICAS DESCRITIVAS:")
print(estatisticas[["mean", "median", "std", "min", "max"]].round(2))

# ==========================================================
# HISTOGRAMAS DAS VARIÁVEIS NUMÉRICAS
# ==========================================================

import matplotlib.pyplot as plt

dados_numericos = dados.select_dtypes(include="number")

fig, eixos = plt.subplots(5, 3, figsize=(16, 18))

eixos = eixos.flatten()

for i, coluna in enumerate(dados_numericos.columns):
    eixos[i].hist(
        dados_numericos[coluna].dropna(),
        bins=30,
        edgecolor="black"
    )

    eixos[i].set_title(coluna, fontsize=11, pad=8)
    eixos[i].set_ylabel("Frequência")

# Remover espaços vazios
for i in range(len(dados_numericos.columns), len(eixos)):
    fig.delaxes(eixos[i])

fig.suptitle(
    "Distribuição das Variáveis do Dataset AirQualityUCI",
    fontsize=16
)

plt.subplots_adjust(
    top=0.94,
    hspace=0.55,
    wspace=0.30
)

# Salvar o gráfico
fig.savefig("graficos/histogramas.png", dpi=300, bbox_inches="tight")

plt.show()

# ==========================================================
# BOXPLOTS DAS VARIÁVEIS NUMÉRICAS
# ==========================================================

fig, eixos = plt.subplots(5, 3, figsize=(16, 18))

eixos = eixos.flatten()

for i, coluna in enumerate(dados_numericos.columns):
    eixos[i].boxplot(
        dados_numericos[coluna].dropna(),
        showfliers=True
    )

    eixos[i].set_title(coluna, fontsize=11, pad=8)
    eixos[i].set_ylabel("Valores")

# Remover espaços vazios
for i in range(len(dados_numericos.columns), len(eixos)):
    fig.delaxes(eixos[i])

fig.suptitle(
    "Boxplots das Variáveis do Dataset AirQualityUCI",
    fontsize=16
)

plt.subplots_adjust(
    top=0.92,
    hspace=0.55,
    wspace=0.30
)

# Salvar o gráfico
fig.savefig("graficos/boxplots.png", dpi=300, bbox_inches="tight")

plt.show()


# ==========================================================
# MATRIZ DE CORRELAÇÃO
# ==========================================================

import seaborn as sns

# Calcular a correlação entre as variáveis numéricas
correlacao = dados_numericos.corr()

# Criar o gráfico
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlacao,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Matriz de Correlação - AirQualityUCI")
plt.tight_layout()

plt.savefig("graficos/matriz_correlacao.png", dpi=300, bbox_inches="tight")

plt.show()


# ==========================================================
# GRÁFICOS DE DISPERSÃO (SCATTER PLOTS)
# ==========================================================

fig, eixos = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1
eixos[0].scatter(
    dados["C6H6(GT)"],
    dados["PT08.S2(NMHC)"],
    alpha=0.4
)
eixos[0].set_title("C6H6(GT) x PT08.S2(NMHC)")
eixos[0].set_xlabel("C6H6(GT)")
eixos[0].set_ylabel("PT08.S2(NMHC)")

# Gráfico 2
eixos[1].scatter(
    dados["CO(GT)"],
    dados["NOx(GT)"],
    alpha=0.4
)
eixos[1].set_title("CO(GT) x NOx(GT)")
eixos[1].set_xlabel("CO(GT)")
eixos[1].set_ylabel("NOx(GT)")

# Gráfico 3
eixos[2].scatter(
    dados["T"],
    dados["RH"],
    alpha=0.4
)
eixos[2].set_title("Temperatura x Umidade")
eixos[2].set_xlabel("Temperatura")
eixos[2].set_ylabel("Umidade relativa")

plt.tight_layout()

# Salvar os três gráficos de dispersão
fig.savefig("graficos/graficos_dispersao.png", dpi=300, bbox_inches="tight")

plt.show()

# ==========================================================
# ETAPA 4 - PRÉ-PROCESSAMENTO PARA KMEANS
# SELEÇÃO DAS VARIÁVEIS
# ==========================================================

# Definir a variável-alvo da regressão
variavel_alvo = "C6H6(GT)"

# Criar conjunto de dados para o KMeans
X_kmeans = dados.drop(
    columns=["Date", "Time", "C6H6(GT)", "NMHC(GT)"]
)

# Exibir as variáveis selecionadas
print("\nVARIÁVEIS SELECIONADAS PARA O KMEANS:")
print(X_kmeans.columns.tolist())

# Exibir dimensões
print("\nDIMENSÕES DOS DADOS PARA KMEANS:")
print(X_kmeans.shape)

# Verificar valores ausentes nas variáveis selecionadas
print("\nVALORES AUSENTES NAS VARIÁVEIS DO KMEANS:")
print(X_kmeans.isnull().sum())

# ==========================================================
# TRATAMENTO DOS VALORES AUSENTES
# ==========================================================

from sklearn.impute import SimpleImputer

# Criar o imputador utilizando a mediana
imputador = SimpleImputer(strategy="median")

# Preencher os valores ausentes
X_kmeans = pd.DataFrame(
    imputador.fit_transform(X_kmeans),
    columns=X_kmeans.columns,
    index=X_kmeans.index
)

# Guardar uma cópia dos dados em suas escalas originais
X_kmeans_original = X_kmeans.copy()

# Verificar se ainda existem valores ausentes
print("\nVALORES AUSENTES APÓS O TRATAMENTO:")
print(X_kmeans.isnull().sum().sum())

# Exibir as dimensões finais
print("\nDIMENSÕES APÓS O TRATAMENTO:")
print(X_kmeans.shape)

# ==========================================================
# PADRONIZAÇÃO DOS DADOS PARA KMEANS
# ==========================================================

from sklearn.preprocessing import StandardScaler

# Criar o padronizador
scaler = StandardScaler()

# Aplicar a padronização
X_kmeans_scaled = scaler.fit_transform(X_kmeans)

# Converter novamente para DataFrame
X_kmeans_scaled = pd.DataFrame(
    X_kmeans_scaled,
    columns=X_kmeans.columns,
    index=X_kmeans.index
)

# Verificar as médias das variáveis padronizadas
print("\nMÉDIAS APÓS A PADRONIZAÇÃO:")
print(X_kmeans_scaled.mean().round(2))

# Verificar os desvios padrão
print("\nDESVIOS PADRÃO APÓS A PADRONIZAÇÃO:")
print(X_kmeans_scaled.std().round(2))

# Exibir dimensões
print("\nDIMENSÕES DOS DADOS PADRONIZADOS:")
print(X_kmeans_scaled.shape)

# ==========================================================
# ETAPA 5 - INVESTIGAÇÃO DO MELHOR VALOR DE K
# MÉTODO DO COTOVELO E ÍNDICE DE SILHUETA
# ==========================================================

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Valores de K que serão avaliados
valores_k = range(2, 11)

# Listas para armazenar os resultados
inercia = []
silhueta = []

# Testar cada valor de K
for k in valores_k:

    print(f"Avaliando K = {k}...")

    # Criar e treinar o modelo
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    rotulos = modelo.fit_predict(X_kmeans_scaled)

    # Armazenar a inércia
    inercia.append(modelo.inertia_)

    # Calcular a silhueta usando uma amostra de 3000 registros
    score = silhouette_score(
        X_kmeans_scaled,
        rotulos,
        sample_size=3000,
        random_state=42
    )

    silhueta.append(score)

# Exibir os resultados
print("\nRESULTADOS DA AVALIAÇÃO DE K:")

for k, iner, sil in zip(valores_k, inercia, silhueta):
    print(f"K = {k} | Inércia = {iner:.2f} | Silhueta = {sil:.4f}")

# Criar os gráficos
fig, eixos = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico do método do cotovelo
eixos[0].plot(valores_k, inercia, marker="o")
eixos[0].set_title("Método do Cotovelo")
eixos[0].set_xlabel("Número de grupos (K)")
eixos[0].set_ylabel("Inércia")
eixos[0].set_xticks(list(valores_k))
eixos[0].grid(True)

# Gráfico do índice de silhueta
eixos[1].plot(valores_k, silhueta, marker="o")
eixos[1].set_title("Índice de Silhueta")
eixos[1].set_xlabel("Número de grupos (K)")
eixos[1].set_ylabel("Silhouette Score")
eixos[1].set_xticks(list(valores_k))
eixos[1].grid(True)

plt.tight_layout()

# Salvar os gráficos do cotovelo e da silhueta
fig.savefig("graficos/metodo_cotovelo_silhueta.png", dpi=300, bbox_inches="tight")

plt.show()


# ==========================================================
# KMEANS FINAL - K = 2
# ==========================================================

# Criar o modelo definitivo
kmeans_final = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Treinar o modelo e identificar os grupos
clusters = kmeans_final.fit_predict(X_kmeans_scaled)

# Adicionar os grupos aos dados originais
X_kmeans_original["Cluster"] = clusters

# Contar os registros em cada grupo
print("\nQUANTIDADE DE REGISTROS POR CLUSTER:")
print(X_kmeans_original["Cluster"].value_counts().sort_index())

# Calcular a porcentagem de cada grupo
print("\nPORCENTAGEM DE REGISTROS POR CLUSTER:")
print(
    (X_kmeans_original["Cluster"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)


# ==========================================================
# ETAPSA 6 - INTERPRETAÇÃO DOS CLUSTERS
# MÉDIA E MEDIANAS POR GRUPO
# ==========================================================

# Calcular médias das variáveis por cluster
medias_clusters = X_kmeans_original.groupby("Cluster").mean()

# Calcular medianas das variáveis por cluster
medianas_clusters = X_kmeans_original.groupby("Cluster").median()

# Exibir todas as colunas das médias
print("\nMÉDIAS DAS VARIÁVEIS POR CLUSTER:")
print(medias_clusters.round(2).to_string())

# Exibir todas as colunas das medianas
print("\nMEDIANAS DAS VARIÁVEIS POR CLUSTER:")
print(medianas_clusters.round(2).to_string())


# ==========================================================
# ETAPA 6 - VISUALIZAÇÃO COMPARATIVA DOS CLUSTERS
# ==========================================================

import matplotlib.pyplot as plt

# Primeiro gráfico: indicadores de poluição
variaveis_poluentes = ["CO(GT)", "NOx(GT)", "NO2(GT)"]

fig, eixos = plt.subplots(1, 3, figsize=(15, 5))

for i, coluna in enumerate(variaveis_poluentes):
    dados_boxplot = [
        X_kmeans_original.loc[
            X_kmeans_original["Cluster"] == cluster, coluna
        ]
        for cluster in [0, 1]
    ]

    eixos[i].boxplot(dados_boxplot, tick_labels=["Cluster 0", "Cluster 1"])
    eixos[i].set_title(coluna)
    eixos[i].set_ylabel("Concentração")
    eixos[i].grid(axis="y", alpha=0.3)

fig.suptitle("Comparação dos Indicadores de Poluição por Cluster")
plt.tight_layout()

fig.savefig("graficos/comparacao_poluicao_clusters.png", dpi=300, bbox_inches="tight")

plt.show()


# Segundo gráfico: variáveis meteorológicas
variaveis_climaticas = ["T", "RH", "AH"]

fig, eixos = plt.subplots(1, 3, figsize=(15, 5))

for i, coluna in enumerate(variaveis_climaticas):
    dados_boxplot = [
        X_kmeans_original.loc[
            X_kmeans_original["Cluster"] == cluster, coluna
        ]
        for cluster in [0, 1]
    ]

    eixos[i].boxplot(dados_boxplot, tick_labels=["Cluster 0", "Cluster 1"])
    eixos[i].set_title(coluna)
    eixos[i].set_ylabel("Valores")
    eixos[i].grid(axis="y", alpha=0.3)

fig.suptitle("Comparação das Variáveis Meteorológicas por Cluster")
plt.tight_layout()

fig.savefig("graficos/comparacao_meteorologica_clusters.png", dpi=300, bbox_inches="tight")

plt.show()

# ==========================================================
# MOMENTO II - APRENDIZADO SUPERVISIONADO
# REGRESSÃO
# ==========================================================

# Definir a variável-alvo
variavel_alvo = "C6H6(GT)"

# Remover registros que não possuem valor na variável-alvo
dados_regressao = dados.dropna(subset=[variavel_alvo]).copy()

# Separar as variáveis preditoras
X_reg = dados_regressao.drop(
    columns=["Date", "Time", "C6H6(GT)", "NMHC(GT)"]
)

# Separar a variável que queremos prever
y_reg = dados_regressao[variavel_alvo]

print("\n===== PREPARAÇÃO DOS DADOS PARA REGRESSÃO =====")
print("Formato das variáveis preditoras:", X_reg.shape)
print("Formato da variável-alvo:", y_reg.shape)
print("Valores ausentes na variável-alvo:", y_reg.isna().sum())

# ==========================================================
# DIVISÃO DOS DADOS EM TREINAMENTO E TESTE
# ==========================================================

from sklearn.model_selection import train_test_split

# Recuperar os clusters correspondentes aos registros da regressão
clusters_reg = X_kmeans_original.loc[dados_regressao.index, "Cluster"]

# Dividir os dados: 80% treinamento e 20% teste
X_treino, X_teste, y_treino, y_teste, clusters_treino, clusters_teste = train_test_split(
    X_reg,
    y_reg,
    clusters_reg,
    test_size=0.20,
    random_state=42
)

print("\n===== DIVISÃO DOS DADOS =====")
print("Dados de treinamento:", X_treino.shape)
print("Dados de teste:", X_teste.shape)
print("Alvos de treinamento:", y_treino.shape)
print("Alvos de teste:", y_teste.shape)


# ==========================================================
# PREPARAÇÃO DOS DADOS PARA OS MODELOS
# ==========================================================

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Criar o processo de tratamento dos dados
preprocessador = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("padronizador", StandardScaler())
])

print("\n===== PREPROCESSAMENTO CONFIGURADO =====")
print("Método para valores ausentes: Mediana")
print("Método de padronização: StandardScaler")
print("Variáveis de treinamento:", X_treino.shape[1])


# ==========================================================
# MOMENTO II - TREINAMENTO DOS TRES MODELOS DE REGRESSAO
# ==========================================================

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold

# ----------------------------------------------------------
# MODELO 1 - REGRESSAO LINEAR
# ----------------------------------------------------------

modelo_linear = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("padronizador", StandardScaler()),
    ("modelo", LinearRegression())
])

modelo_linear.fit(X_treino, y_treino)

previsoes_linear = modelo_linear.predict(X_teste)

print("\n===== REGRESSAO LINEAR =====")
print("Modelo treinado com sucesso!")


# ----------------------------------------------------------
# MODELO 2 - RIDGE (REGULARIZACAO L2)
# ----------------------------------------------------------

pipeline_ridge = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("padronizador", StandardScaler()),
    ("modelo", Ridge())
])

# Valores de Alpha que serao testados
parametros_ridge = {
    "modelo__alpha": [0.01, 0.1, 1, 10, 100]
}

# Validacao cruzada com 5 partes
validacao = KFold(n_splits=5, shuffle=True, random_state=42)

busca_ridge = GridSearchCV(
    pipeline_ridge,
    parametros_ridge,
    cv=validacao,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

busca_ridge.fit(X_treino, y_treino)

modelo_ridge = busca_ridge.best_estimator_

previsoes_ridge = modelo_ridge.predict(X_teste)

print("\n===== RIDGE =====")
print("Melhor Alpha:", busca_ridge.best_params_)
print("Melhor resultado da validacao cruzada:",
      -busca_ridge.best_score_)


# ----------------------------------------------------------
# MODELO 3 - LASSO (REGULARIZACAO L1)
# ----------------------------------------------------------

pipeline_lasso = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("padronizador", StandardScaler()),
    ("modelo", Lasso(max_iter=10000))
])

# Valores de Alpha que serao testados
parametros_lasso = {
    "modelo__alpha": [0.01, 0.1, 1, 10, 100]
}

busca_lasso = GridSearchCV(
    pipeline_lasso,
    parametros_lasso,
    cv=validacao,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

busca_lasso.fit(X_treino, y_treino)

modelo_lasso = busca_lasso.best_estimator_

previsoes_lasso = modelo_lasso.predict(X_teste)

print("\n===== LASSO =====")
print("Melhor Alpha:", busca_lasso.best_params_)
print("Melhor resultado da validacao cruzada:",
      -busca_lasso.best_score_)


# ----------------------------------------------------------
# RESUMO DO TREINAMENTO
# ----------------------------------------------------------

print("\n===== RESUMO DOS TREINAMENTOS =====")
print("Regressao Linear: concluida")
print("Ridge: concluido")
print("Lasso: concluido")

print("\nQuantidade de previsoes por modelo:")
print("Linear:", len(previsoes_linear))
print("Ridge:", len(previsoes_ridge))
print("Lasso:", len(previsoes_lasso))



# ==========================================================
# AVALIACAO DOS TRES MODELOS DE REGRESSAO
# ==========================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Organizar as previsoes dos modelos
resultados_modelos = {
    "Regressao Linear": previsoes_linear,
    "Ridge": previsoes_ridge,
    "Lasso": previsoes_lasso
}

# Lista para armazenar as metricas
metricas = []

for nome, previsoes in resultados_modelos.items():

    mae = mean_absolute_error(y_teste, previsoes)
    mse = mean_squared_error(y_teste, previsoes)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_teste, previsoes)

    metricas.append({
        "Modelo": nome,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# Criar tabela comparativa
tabela_metricas = pd.DataFrame(metricas)

print("\n===== COMPARACAO DAS METRICAS =====")
print(tabela_metricas.to_string(index=False))

# Identificar o modelo com menor RMSE
melhor_modelo = tabela_metricas.loc[
    tabela_metricas["RMSE"].idxmin()
]

print("\n===== MELHOR MODELO PELO RMSE =====")
print("Modelo:", melhor_modelo["Modelo"])
print("RMSE:", melhor_modelo["RMSE"])


# ==========================================================
# GRAFICOS DE AVALIACAO DOS MODELOS
# ==========================================================

import matplotlib.pyplot as plt

# ----------------------------------------------------------
# GRAFICO 1 - VALORES REAIS X PREVISTOS
# ----------------------------------------------------------

fig, eixos = plt.subplots(1, 3, figsize=(18, 5))

for i, (nome, previsoes) in enumerate(resultados_modelos.items()):

    eixos[i].scatter(y_teste, previsoes, alpha=0.4)

    # Linha ideal: previsao igual ao valor real
    minimo = min(y_teste.min(), previsoes.min())
    maximo = max(y_teste.max(), previsoes.max())

    eixos[i].plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--"
    )

    eixos[i].set_title(nome)
    eixos[i].set_xlabel("Valores reais")
    eixos[i].set_ylabel("Valores previstos")

plt.suptitle("Comparacao entre valores reais e previstos")
plt.tight_layout()

fig.savefig("graficos/valores_reais_vs_previstos.png", dpi=300, bbox_inches="tight")

plt.show()



# ----------------------------------------------------------
# GRAFICO 2 - DISTRIBUICAO DOS RESIDUOS
# ----------------------------------------------------------

fig, eixos = plt.subplots(1, 3, figsize=(18, 5))

for i, (nome, previsoes) in enumerate(resultados_modelos.items()):

    residuos = y_teste.values - previsoes

    eixos[i].hist(residuos, bins=30, edgecolor="black")
    eixos[i].axvline(0, linestyle="--")

    eixos[i].set_title(nome)
    eixos[i].set_xlabel("Residuos")
    eixos[i].set_ylabel("Frequencia")

plt.suptitle("Distribuicao dos residuos dos modelos")
plt.tight_layout()
fig.savefig("graficos/distribuicao_residuos.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# COMPARACAO DOS COEFICIENTES DOS MODELOS
# ==========================================================

# Obter os coeficientes de cada modelo
coef_linear = modelo_linear.named_steps["modelo"].coef_
coef_ridge = modelo_ridge.named_steps["modelo"].coef_
coef_lasso = modelo_lasso.named_steps["modelo"].coef_

# Criar tabela comparativa
tabela_coeficientes = pd.DataFrame({
    "Variavel": X_treino.columns,
    "Linear": coef_linear,
    "Ridge": coef_ridge,
    "Lasso": coef_lasso
})

print("\n===== COMPARACAO DOS COEFICIENTES =====")
print(tabela_coeficientes.to_string(index=False))

# Verificar quantas variaveis o Lasso eliminou
variaveis_eliminadas = np.sum(coef_lasso == 0)

print("\n===== ANALISE DO LASSO =====")
print("Quantidade de coeficientes iguais a zero:", variaveis_eliminadas)
print("Quantidade de variaveis utilizadas:", len(coef_lasso))


# ==========================================================
# ANALISE DA VARIAVEL-ALVO POR CLUSTER
# ==========================================================

dados_analise_clusters = pd.DataFrame({
    "C6H6(GT)": y_reg,
    "Cluster": clusters_reg
})

print("\n===== DISTRIBUICAO DE C6H6(GT) POR CLUSTER =====")

resumo_clusters = dados_analise_clusters.groupby("Cluster")["C6H6(GT)"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(resumo_clusters)

# Grafico da distribuicao da variavel-alvo por cluster


dados_analise_clusters.boxplot(
    column="C6H6(GT)",
    by="Cluster"
)

plt.title("Distribuicao de C6H6(GT) por Cluster")
plt.suptitle("")
plt.xlabel("Cluster")
plt.ylabel("C6H6(GT)")
plt.grid(axis="y", alpha=0.3)
plt.savefig("graficos/distribuicao_C6H6_por_cluster.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# MAE E RMSE POR CLUSTER NO CONJUNTO DE TESTE
# ==========================================================

resultados_por_cluster = []

for cluster in sorted(clusters_teste.unique()):

    mascara = clusters_teste.to_numpy() == cluster

    y_real_cluster = y_teste.iloc[np.where(mascara)[0]]

    for nome, previsoes in resultados_modelos.items():

        previsoes_cluster = previsoes[mascara]

        mae = mean_absolute_error(
            y_real_cluster,
            previsoes_cluster
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_real_cluster,
                previsoes_cluster
            )
        )

        resultados_por_cluster.append({
            "Cluster": cluster,
            "Modelo": nome,
            "Quantidade": mascara.sum(),
            "MAE": mae,
            "RMSE": rmse
        })

tabela_erros_cluster = pd.DataFrame(resultados_por_cluster)

print("\n===== ERROS DOS MODELOS POR CLUSTER =====")
print(tabela_erros_cluster.to_string(index=False))