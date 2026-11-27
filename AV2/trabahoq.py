# Dataset: Student Performance (UCI, id=320) | Alvo: G3
# ==============================================================================

# Importa bibliotecas básicas para cálculo numérico, tabelas e gráficos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Importa função para baixar a base de dados diretamente do repositório da UCI
from ucimlrepo import fetch_ucirepo

# Importa ferramentas nativas do scikit-learn para pré-processamento e agrupamento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


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

# 3a) Exibe quantidade de linhas e colunas da tabela e informações de tipos de dados
print("Dimensões (amostras, colunas):", tabela_alunos.shape)
print("info", tabela_alunos.info())

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

# Adiciona o cluster encontrado e traz G3 de volta apenas para análise posterior (fora do K-Means)
tabela_clusters_original["cluster"] = clusters_finais
tabela_clusters_original["G3"] = tabela_alunos_limpa["G3"]

# Lista todas as colunas numéricas da tabela (agora incluindo G3), exceto 'cluster'
colunas_numericas_cluster = [
    coluna for coluna in tabela_clusters_original.select_dtypes(include=np.number).columns
    if coluna != "cluster"
]

# 6a e 6b) Calcula média e mediana de cada variável numérica por cluster na escala original (incluindo G3)
medias_por_cluster = tabela_clusters_original.groupby("cluster")[colunas_numericas_cluster].mean().round(2).T
medianas_por_cluster = tabela_clusters_original.groupby("cluster")[colunas_numericas_cluster].median().round(2).T

print("\n6a/6b) Médias por cluster (escala original):\n", medias_por_cluster)
print("\n6a/6b) Medianas por cluster (escala original):\n", medianas_por_cluster)

# 6c) Compara visualmente faltas, tempo de estudo e nota final (G3) entre os clusters via boxplots
figura_perfis, eixos_perfis = plt.subplots(1, 3, figsize=(16, 4.5))

sns.boxplot(x="cluster", y="absences", data=tabela_clusters_original, ax=eixos_perfis[0])
eixos_perfis[0].set_title("6c) Faltas (absences) por cluster")

sns.boxplot(x="cluster", y="studytime", data=tabela_clusters_original, ax=eixos_perfis[1])
eixos_perfis[1].set_title("6c) Tempo de estudo (studytime) por cluster")

sns.boxplot(x="cluster", y="G3", data=tabela_clusters_original, ax=eixos_perfis[2])
eixos_perfis[2].set_title("6c) Nota final (G3) por cluster")

plt.tight_layout()
plt.show()

# 6d) Principais diferenças entre grupos: consumo de álcool, tempo de estudo, escolaridade dos pais, faltas e G3.
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

#------------------------------------------------------------------------------
# 8. Preparação dos dados para regressão
# ------------------------------------------------------------------------------
print("\n--- 8. Preparação dos dados para regressão ---")

# 8a)Dividir o dataset em conjuntos de treinamento e teste, utilizando uma proporção de 80% e 20%.
X = tabela_alunos.drop(columns=["G1", "G2", "G3"])
y = tabela_alunos["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# 8b) Aplicar o tratamento necessário às características categóricas e numéricas.
# 8c) Padronizar as características preditoras, especialmente para os modelos Ridge e Lasso.
colunas_numericas = X_train.select_dtypes(include=['number']).columns.tolist()
colunas_categoricas = X_train.select_dtypes(exclude=['number']).columns.tolist()

pipeline_numerico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='median')),
    ('escalonador', StandardScaler())  # Item 8c: Padronização para Ridge e Lasso
])

pipeline_categorico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Item 8b: Categóricas
])

preprocessador = ColumnTransformer(
    transformers=[
        ('num', pipeline_numerico, colunas_numericas),
        ('cat', pipeline_categorico, colunas_categoricas)
    ]
)
X_train_tratado = preprocessador.fit_transform(X_train)
X_test_tratado = preprocessador.transform(X_test)

# 8d) Garantir que as transformações sejam ajustadas somente com os dados de treinamento e posteriormente aplicadas ao conjunto de teste.
# 8e) Utilizar a mesma divisão de dados para os três modelos, garantindo condições equivalentes de comparação.

from sklearn.model_selection import GridSearchCV, KFold

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

modelos = {

    # 9a) Treinar um modelo de regressão linear convencional.

    "Regressão Linear": Pipeline([
        ("prep", preprocessador),
        ("reg", LinearRegression())
    ]),

    # 9b) Treinar um modelo Ridge, avaliando diferentes valores do hiperparâmetro \(\alpha\).

    "Ridge": GridSearchCV(
        Pipeline([
            ("prep", preprocessador),
            ("reg", Ridge())
        ]),
        param_grid={"reg__alpha": alphas},
        cv=kfold,
        scoring="r2"
    ),

    # 9c) Treinar um modelo Lasso, avaliando diferentes valores do hiperparâmetro \(\alpha\).

    "Lasso": GridSearchCV(
        Pipeline([
            ("prep", preprocessador),
            ("reg", Lasso(max_iter=100000))
        ]),
        param_grid={"reg__alpha": alphas},
        cv=kfold,
        scoring="r2"
    )
}

for nome, modelo in modelos.items():

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    if hasattr(modelo, "best_params_"):
        print(nome, "melhor alpha:", modelo.best_params_["reg__alpha"])

    print(
        f"{nome} -> "
        f"MSE: {mean_squared_error(y_test, y_pred):.2f} | "
        f"R²: {r2_score(y_test, y_pred):.4f}"
    )

# ------------------------------------------------------------------------------
# 9. Treinamento dos modelos de regressão
# ------------------------------------------------------------------------------
print("\n--- 9. Treinamento dos modelos de regressão ---")

# 9e) Registrar as configurações utilizadas nos três modelos
print("\n--- [9e] CONFIGURAÇÕES DOS MODELOS ---")
for nome, modelo in modelos.items():
    if hasattr(modelo, "best_params_"):
        print(f"{nome}: Melhor Alpha = {modelo.best_params_['reg__alpha']} | Config = {modelo.best_estimator_.named_steps['reg'].get_params()}")
    else:
        print(f"{nome}: Config = {modelo.named_steps['reg'].get_params()}")

# 9f) Comparar os coeficientes obtidos pelos modelos

lr_final = modelos["Regressão Linear"]
ridge_final = modelos["Ridge"].best_estimator_
lasso_final = modelos["Lasso"].best_estimator_

nomes_features = lr_final.named_steps["prep"].get_feature_names_out()

df_coefs = pd.DataFrame({
    "Característica": nomes_features,
    "Linear": lr_final.named_steps["reg"].coef_,
    "Ridge": ridge_final.named_steps["reg"].coef_,
    "Lasso": lasso_final.named_steps["reg"].coef_
})

zeros_lasso = (df_coefs["Lasso"] == 0).sum()
print(f"\n--- [9f] COMPARAÇÃO DE COEFICIENTES (Zerados no Lasso: {zeros_lasso} de {len(nomes_features)}) ---")
print(df_coefs.to_string(index=False))

# Extrai os resultados dos modelos Ridge e Lasso do GridSearch
resultados_ridge = modelos["Ridge"].cv_results_
resultados_lasso = modelos["Lasso"].cv_results_

# Pega os valores de alpha testados (convertidos para string ou float para o eixo X)
alphas_testados = modelos["Ridge"].param_grid["reg__alpha"]

# Pega a pontuação média (mean_test_score) de cada alpha na validação cruzada
scores_ridge = resultados_ridge["mean_test_score"]
scores_lasso = resultados_lasso["mean_test_score"]

# Configuração do Gráfico
plt.figure(figsize=(9, 5))

# Plotando as curvas (usamos escala logarítmica no eixo X porque os alphas variam muito de magnitude, ex: 0.001 a 100)
plt.plot(alphas_testados, scores_ridge, marker='o', linestyle='-', label='Ridge (L2)', color='#DD8452', linewidth=2)
plt.plot(alphas_testados, scores_lasso, marker='s', linestyle='--', label='Lasso (L1)', color='#55A868', linewidth=2)

# Personalização
plt.xscale('log') # Essencial para alphas em ordens de grandeza diferentes
plt.xlabel('Valor de Alpha ($alpha$) [Escala Log]', fontsize=11)
plt.ylabel('Score Médio ($R^2$ na Validação Cruzada)', fontsize=11)
plt.title('Performance dos Modelos por Valor de Alpha (Grid Search K-Fold)', fontsize=13, fontweight='bold', pad=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()

# 1. Copia a tabela e limpa os nomes técnicos (tira 'num__' e 'cat__')
df_slide = df_coefs.copy()
df_slide["Característica"] = df_slide["Característica"].str.replace("num__|cat__", "", regex=True)

# 2. Pega apenas as 10 variáveis de maior impacto (pelo peso absoluto na Linear)
top_10 = df_slide.reindex(df_slide["Linear"].abs().sort_values(ascending=True).index).tail(10)

# 3. Plota o gráfico limpo para o slide
ax = top_10.plot(
    x="Característica",
    y=["Linear", "Ridge", "Lasso"],
    kind="barh",
    figsize=(10, 6),
    color=["#4C72B0", "#DD8452", "#55A868"], # Cores suaves e profissionais
    width=0.75
)

# 4. Deixa o visual minimalista (sem bordas desnecessárias)
plt.title("Top 10 Variáveis: Impacto da Regularização nos Coeficientes", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Peso (Coeficiente)", fontsize=11)
plt.ylabel("") # Remove o título "Característica" pois já é óbvio
plt.axvline(0, color="black", linewidth=0.8) # Linha central no zero para destacar positivos/negativos
plt.grid(axis="x", linestyle="--", alpha=0.4)

# Remove as bordas de cima e da direita
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 10. Avaliação e comparação dos modelos
# ------------------------------------------------------------------------------
print("\n--- 10. Avaliação e comparação dos modelos ---")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Calcula as métricas para cada modelo e guarda na lista
resultados_metricas = []

for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    resultados_metricas.append({
        "Modelo": nome,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    })

# 2. Cria o DataFrame com as métricas (agora a variável 'df_metricas' existe!)
df_metricas = pd.DataFrame(resultados_metricas)

# Exibe também no console por segurança
print("\n--- TABELA COMPARATIVA DE MÉTRICAS ---")
print(df_metricas.round(4).to_string(index=False))

# 3. Plota a tabela usando o matplotlib (plt)
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis('tight')
ax.axis('off')

dados_tabela = df_metricas.round(4).values
colunas = df_metricas.columns

tabela = ax.table(
    cellText=dados_tabela,
    colLabels=colunas,
    cellLoc='center',
    loc='center'
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.scale(1.2, 1.5)

plt.title('Tabela Comparativa de Métricas (Linear, Ridge e Lasso)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
cores = ['#4C72B0', '#DD8452', '#55A868']

for i, (nome, modelo) in enumerate(modelos.items()):
    y_pred = modelo.predict(X_test)

    # Scatter plot
    axes[i].scatter(y_test, y_pred, alpha=0.6, color=cores[i], edgecolor='k', s=40)

    # Linha ideal de 45 graus (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Previsão Perfeita')

    axes[i].set_title(f"{nome}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Valores Reais", fontsize=10)
    if i == 0:
        axes[i].set_ylabel("Valores Previstos", fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[i].legend(loc='upper left', fontsize=8)

plt.suptitle("Comparação: Valores Reais vs. Previstos", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, (nome, modelo) in enumerate(modelos.items()):
    y_pred = modelo.predict(X_test)
    residuos = y_test - y_pred

    # Histograma dos resíduos
    axes[i].hist(residuos, bins=20, color=cores[i], edgecolor='black', alpha=0.7)
    axes[i].axvline(0, color='red', linestyle='--', linewidth=2, label='Erro Zero')

    axes[i].set_title(f"Resíduos - {nome}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Erro (Real - Previsto)", fontsize=10)
    if i == 0:
        axes[i].set_ylabel("Frequência", fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[i].legend(fontsize=8)

plt.suptitle("Análise da Distribuição dos Resíduos", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()