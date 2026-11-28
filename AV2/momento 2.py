import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Configurações dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

df = pd.read_csv('/content/drive/MyDrive/2026.2/ia/AV 2/dataset/hour.csv')

"""CODIGOS DA GABI"""

# ==========================================
# 1. CARREGAMENTO DO DATASET
# ==========================================

#df = pd.read_csv("dataset/hour.csv")

# Criar pasta para armazenar os resultados
os.makedirs("resultados", exist_ok=True)

print("==========================================")
print("DEFINICAO DO PROBLEMA DE REGRESSAO")
print("==========================================")

print("\nDimensoes do dataset:")
print("Amostras:", df.shape[0])
print("Caracteristicas:", df.shape[1])


# ==========================================
# 2. SELECAO DA VARIAVEL DEPENDENTE (ALVO)
# ==========================================

# A) Definir a variavel que sera prevista

alvo = "cnt"

print("\nVariavel dependente selecionada:", alvo)

print("\nJustificativa:")
print("A variavel cnt representa a quantidade total de")
print("bicicletas alugadas por hora, sendo adequada")
print("para um problema de regressao.")


# ==========================================
# 3. IDENTIFICACAO DAS VARIAVEIS INDEPENDENTES
# ==========================================

# B) Selecionar as caracteristicas preditoras

caracteristicas = [
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed"
]

X = df[caracteristicas].copy()
y = df[alvo].copy()

print("\nVariaveis independentes selecionadas:")
print(caracteristicas)

print("\nQuantidade de variaveis preditoras:", len(caracteristicas))

print("\nPrimeiras linhas das variaveis preditoras:")
print(X.head())

print("\nPrimeiros valores da variavel alvo:")
print(y.head())


# ==========================================
# 4. IDENTIFICACAO DE POSSIVEIS VAZAMENTOS
# ==========================================

# E) Verificar a relacao entre cnt, casual e registered

print("\n==========================================")
print("VERIFICACAO DE VAZAMENTO DE DADOS")
print("==========================================")

inconsistencias = (
    df["cnt"] != df["casual"] + df["registered"]
).sum()

print("\nRegistros inconsistentes na soma:")
print(inconsistencias)

print("\nRelacao entre as variaveis:")
print("cnt = casual + registered")

print("\nAs variaveis casual e registered nao serao")
print("utilizadas como preditoras, pois revelam")
print("diretamente a quantidade total de alugueis.")

# Verificar se o alvo esta entre os preditores
if alvo in caracteristicas:
    raise ValueError("A variavel alvo nao pode estar entre os preditores.")

print("\nVariavel alvo excluida dos preditores corretamente.")


# ==========================================
# 5. ANALISE DA RELACAO ENTRE AS VARIAVEIS
# ==========================================

# C) Calcular correlacao entre os preditores e o alvo

dados_analise = df[caracteristicas + [alvo]]

correlacao = dados_analise.corr(numeric_only=True)

correlacao_alvo = (
    correlacao[alvo]
    .drop(alvo)
    .sort_values(ascending=False)
)

print("\n==========================================")
print("CORRELACAO COM A VARIAVEL ALVO")
print("==========================================")

print(correlacao_alvo.round(3))


# Grafico de correlacao com a variavel alvo

plt.figure(figsize=(10, 6))

sns.barplot(
    x=correlacao_alvo.values,
    y=correlacao_alvo.index
)

plt.title("Correlacao das variaveis com cnt")
plt.xlabel("Coeficiente de correlacao")
plt.ylabel("Variaveis")

plt.tight_layout()
plt.savefig("resultados/correlacao_com_alvo.png")
plt.show()
plt.close()


# ==========================================
# 6. GRAFICOS DE RELACAO COM A VARIAVEL ALVO
# ==========================================

# Relacao entre temperatura e quantidade de alugueis

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="temp",
    y="cnt",
    alpha=0.3
)

plt.title("Temperatura x Quantidade de bicicletas alugadas")
plt.xlabel("Temperatura")
plt.ylabel("Quantidade de bicicletas")

plt.tight_layout()
plt.savefig("resultados/temperatura_cnt.png")
plt.show()
plt.close()


# Relacao entre umidade e quantidade de alugueis

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="hum",
    y="cnt",
    alpha=0.3
)

plt.title("Umidade x Quantidade de bicicletas alugadas")
plt.xlabel("Umidade")
plt.ylabel("Quantidade de bicicletas")

plt.tight_layout()
plt.savefig("resultados/umidade_cnt.png")
plt.show()
plt.close()


# Relacao entre horario e quantidade de alugueis

media_hora = df.groupby("hr")["cnt"].mean()

plt.figure(figsize=(10, 5))

sns.lineplot(
    x=media_hora.index,
    y=media_hora.values,
    marker="o"
)

plt.title("Media de bicicletas alugadas por hora")
plt.xlabel("Hora do dia")
plt.ylabel("Media de alugueis")

plt.xticks(range(24))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("resultados/horario_cnt.png")
plt.show()
plt.close()


# ==========================================
# 7. ANALISE DE CORRELACAO ENTRE PREDITORES
# ==========================================

# D) Identificar correlacoes elevadas

correlacao_preditores = X.corr(numeric_only=True)

print("\n==========================================")
print("CORRELACAO ENTRE VARIAVEIS INDEPENDENTES")
print("==========================================")

print(correlacao_preditores.round(3))


# Verificar pares com correlacao igual ou superior a 0.80

print("\nPares de variaveis com correlacao elevada:")

encontrou = False

colunas = correlacao_preditores.columns

for i in range(len(colunas)):

    for j in range(i + 1, len(colunas)):

        valor = correlacao_preditores.iloc[i, j]

        if abs(valor) >= 0.80:

            encontrou = True

            print(
                colunas[i],
                "e",
                colunas[j],
                ":",
                round(valor, 3)
            )

if not encontrou:
    print("Nenhum par com correlacao elevada foi encontrado.")


# ==========================================
# 8. MATRIZ DE CORRELACAO DOS PREDITORES
# ==========================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlacao_preditores,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Matriz de correlacao das variaveis preditoras")

plt.tight_layout()
plt.savefig("resultados/matriz_correlacao_preditores.png")
plt.show()
plt.close()


# ==========================================
# 9. REGISTRO DAS VARIAVEIS SELECIONADAS
# ==========================================

print("\n==========================================")
print("RESUMO DA DEFINICAO DO PROBLEMA")
print("==========================================")

print("\nVariavel dependente:")
print(alvo)

print("\nVariaveis independentes:")
for coluna in caracteristicas:
    print("-", coluna)

print("\nVariaveis excluidas:")

print("- instant: identificador sequencial.")
print("- dteday: data, com informacoes temporais")
print("  representadas pelas demais variaveis.")
print("- season: redundancia com mnth.")
print("- atemp: alta correlacao com temp.")
print("- casual: vazamento de dados.")
print("- registered: vazamento de dados.")

print("\nDefinicao do problema de regressao concluida!")

print("Graficos salvos na pasta resultados.")

# ==========================================
# 1. CONFIGURACOES INICIAIS
# ==========================================

# No Google Colab, definimos o caminho absoluto do arquivo no Drive
#CAMINHO_DATASET = Path('/content/drive/MyDrive/2026.2/ia/AV 2/dataset/hour.csv')

# Caso o arquivo não esteja no Drive, tentamos a pasta local
#if not CAMINHO_DATASET.exists():
# CAMINHO_DATASET = Path('dataset/hour.csv')

#df = pd.read_csv(CAMINHO_DATASET)

print("==========================================")
print("PRE-PROCESSAMENTO DOS DADOS")
print("==========================================")

print("\nDataset carregado com sucesso!")
print("Dimensoes:", df.shape)


# ==========================================
# 2. DEFINICAO DO ALVO E DOS PREDITORES
# ==========================================

ALVO = "cnt"

CARACTERISTICAS = [
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed"
]

X = df[CARACTERISTICAS].copy()
y = df[ALVO].copy()

print("\nVariavel alvo:", ALVO)
print("Quantidade de preditores:", len(CARACTERISTICAS))


# ==========================================
# 3. IDENTIFICACAO DOS TIPOS DE VARIAVEIS
# ==========================================

# Variaveis categoricas:
# Sao categorias, mesmo quando representadas por numeros.

VARIAVEIS_CATEGORICAS = [
    "mnth",
    "hr",
    "weekday",
    "weathersit"
]

# Variaveis binarias:
# Possuem valores 0 e 1.

VARIAVEIS_BINARIAS = [
    "yr",
    "holiday",
    "workingday"
]

# Variaveis numericas continuas.

VARIAVEIS_NUMERICAS = [
    "temp",
    "hum",
    "windspeed"
]

print("\nVariaveis categoricas:")
print(VARIAVEIS_CATEGORICAS)

print("\nVariaveis binarias:")
print(VARIAVEIS_BINARIAS)

print("\nVariaveis numericas:")
print(VARIAVEIS_NUMERICAS)


# ==========================================
# 4. DIVISAO ENTRE TREINAMENTO E TESTE
# ==========================================

# 80% dos dados para treinamento
# 20% dos dados para teste

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n==========================================")
print("DIVISAO DOS DADOS")
print("==========================================")

print("\nDados originais:", X.shape)
print("Dados de treinamento:", X_train.shape)
print("Dados de teste:", X_test.shape)

print("\nAlvo de treinamento:", y_train.shape)
print("Alvo de teste:", y_test.shape)

print("\nPercentual de treinamento: 80%")
print("Percentual de teste: 20%")


# ==========================================
# 5. CRIACAO DO PRE-PROCESSADOR
# ==========================================

def criar_preprocessador():

    # Codificar variaveis categoricas
    codificador = OneHotEncoder(
        handle_unknown="ignore"
    )

    # Organizar o tratamento de cada grupo de variaveis
    transformador_colunas = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                codificador,
                VARIAVEIS_CATEGORICAS
            ),
            (
                "binarias",
                "passthrough",
                VARIAVEIS_BINARIAS
            ),
            (
                "numericas",
                "passthrough",
                VARIAVEIS_NUMERICAS
            )
        ]
    )

    # Padronizar todas as caracteristicas transformadas
    preprocessador = Pipeline(
        steps=[
            (
                "transformacao",
                transformador_colunas
            ),
            (
                "padronizacao",
                StandardScaler(with_mean=False)
            )
        ]
    )

    return preprocessador


# ==========================================
# 6. AJUSTE E APLICACAO DO PRE-PROCESSAMENTO
# ==========================================

preprocessador = criar_preprocessador()

# O ajuste (fit) e realizado somente nos dados de treinamento.
# Depois, a mesma transformacao e aplicada aos dados de teste.

X_train_processado = preprocessador.fit_transform(X_train)

X_test_processado = preprocessador.transform(X_test)

print("\n==========================================")
print("RESULTADO DO PRE-PROCESSAMENTO")
print("==========================================")

print("\nDimensoes apos o pre-processamento:")
print("Treinamento:", X_train_processado.shape)
print("Teste:", X_test_processado.shape)

print("\nCodificacao das variaveis categoricas: concluida.")
print("Padronizacao das caracteristicas: concluida.")

print("\nO ajuste do preprocessador foi feito somente")
print("com os dados de treinamento.")

print("Os dados de teste receberam apenas a transformacao.")


# ==========================================
# 7. VERIFICACAO DA DIVISAO
# ==========================================

print("\n==========================================")
print("RESUMO DA SECAO 8")
print("==========================================")

print("\nVariavel alvo:", ALVO)

print("Divisao dos dados: 80% treinamento e 20% teste.")

print("Estado aleatorio (random_state): 42")

print("Pre-processamento das variaveis: concluido.")

print("\nPreparacao dos dados para regressao concluida!")

# ==========================================
# 1. IMPORTACAO DOS DADOS E DO PRE-PROCESSADOR
# ==========================================

# Em notebooks, as variáveis e funções declaradas em células executadas anteriormente
# permanecem salvas na memória global do kernel. Usaremos X_train, X_test, y_train,
# y_test e a função criar_preprocessador diretamente daqui.

print("\n==========================================")
print("SECAO 9 - MODELOS DE REGRESSAO")
print("==========================================")

# Todos os modelos utilizarao a mesma divisao
# de treinamento e teste da Secao 8.


# ==========================================
# 2. CONFIGURACAO DA VALIDACAO CRUZADA
# ==========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Valores de alpha que serao testados

valores_alpha = [0.01, 0.1, 1, 10, 100]

print("\nValores de alpha testados:")
print(valores_alpha)


# ==========================================
# 3. MODELO DE REGRESSAO LINEAR
# ==========================================

print("\n==========================================")
print("MODELO 1 - REGRESSAO LINEAR")
print("==========================================")

modelo_linear = Pipeline(
    steps=[
        ("preprocessamento", criar_preprocessador()),
        ("modelo", LinearRegression())
    ]
)

modelo_linear.fit(X_train, y_train)

previsoes_linear = modelo_linear.predict(X_test)

print("\nRegressao Linear treinada com sucesso!")


# ==========================================
# 4. MODELO RIDGE COM GRIDSEARCHCV
# ==========================================

print("\n==========================================")
print("MODELO 2 - RIDGE")
print("==========================================")

modelo_ridge = Pipeline(
    steps=[
        ("preprocessamento", criar_preprocessador()),
        ("modelo", Ridge())
    ]
)

parametros_ridge = {
    "modelo__alpha": valores_alpha
}

busca_ridge = GridSearchCV(
    estimator=modelo_ridge,
    param_grid=parametros_ridge,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

busca_ridge.fit(X_train, y_train)

melhor_ridge = busca_ridge.best_estimator_

previsoes_ridge = melhor_ridge.predict(X_test)

print("\nMelhor configuracao Ridge:")
print(busca_ridge.best_params_)

print("\nMelhor resultado medio na validacao cruzada (RMSE):")

melhor_rmse_ridge = np.sqrt(-busca_ridge.best_score_)

print(round(melhor_rmse_ridge, 3))


# ==========================================
# 5. MODELO LASSO COM GRIDSEARCHCV
# ==========================================

print("\n==========================================")
print("MODELO 3 - LASSO")
print("==========================================")

modelo_lasso = Pipeline(
    steps=[
        ("preprocessamento", criar_preprocessador()),
        ("modelo", Lasso(max_iter=20000))
    ]
)

parametros_lasso = {
    "modelo__alpha": valores_alpha
}

busca_lasso = GridSearchCV(
    estimator=modelo_lasso,
    param_grid=parametros_lasso,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

busca_lasso.fit(X_train, y_train)

melhor_lasso = busca_lasso.best_estimator_

previsoes_lasso = melhor_lasso.predict(X_test)

print("\nMelhor configuracao Lasso:")
print(busca_lasso.best_params_)

print("\nMelhor resultado medio na validacao cruzada (RMSE):")

melhor_rmse_lasso = np.sqrt(-busca_lasso.best_score_)

print(round(melhor_rmse_lasso, 3))


# ==========================================
# 6. AVALIACAO DOS MODELOS
# ==========================================

print("\n==========================================")
print("COMPARACAO DOS MODELOS")
print("==========================================")

def avaliar_modelo(nome, y_real, y_previsto):

    mae = mean_absolute_error(y_real, y_previsto)

    rmse = np.sqrt(
        mean_squared_error(y_real, y_previsto)
    )

    r2 = r2_score(y_real, y_previsto)

    return {
        "Modelo": nome,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


resultados = []

resultados.append(
    avaliar_modelo(
        "Regressao Linear",
        y_test,
        previsoes_linear
    )
)

resultados.append(
    avaliar_modelo(
        "Ridge",
        y_test,
        previsoes_ridge
    )
)

resultados.append(
    avaliar_modelo(
        "Lasso",
        y_test,
        previsoes_lasso
    )
)

tabela_resultados = pd.DataFrame(resultados)
print("\nMetricas no conjunto de teste:")
print(tabela_resultados.round(3).to_string(index=False))



# ==========================================
# 7. COMPARACAO DOS COEFICIENTES
# ==========================================

print("\n==========================================")
print("COMPARACAO DOS COEFICIENTES")
print("==========================================")

# Recuperar os nomes das caracteristicas
# depois da codificacao OneHotEncoder.

preprocessador_linear = (
    modelo_linear.named_steps["preprocessamento"]
)

nomes_caracteristicas = (
    preprocessador_linear
    .get_feature_names_out()
)

coeficientes = pd.DataFrame({
    "Caracteristica": nomes_caracteristicas,
    "Linear": modelo_linear.named_steps["modelo"].coef_,
    "Ridge": melhor_ridge.named_steps["modelo"].coef_,
    "Lasso": melhor_lasso.named_steps["modelo"].coef_
})

print("\nPrimeiros coeficientes:")
print(coeficientes.head(15).round(4).to_string(index=False))


# ==========================================
# 8. IDENTIFICACAO DOS COEFICIENTES ZERADOS
# ==========================================

print("\n==========================================")
print("COEFICIENTES ZERADOS PELO LASSO")
print("==========================================")

coef_lasso = melhor_lasso.named_steps["modelo"].coef_

quantidade_zerados = np.sum(
    np.isclose(coef_lasso, 0, atol=1e-8)
)

quantidade_total = len(coef_lasso)

print("\nQuantidade de coeficientes zerados:", quantidade_zerados)

print("Quantidade total de coeficientes:", quantidade_total)

print(
    "Percentual de coeficientes zerados:",
    round(100 * quantidade_zerados / quantidade_total, 2),
    "%"
)

print("\nCaracteristicas com coeficiente zerado:")

for nome, coef in zip(nomes_caracteristicas, coef_lasso):

    if np.isclose(coef, 0, atol=1e-8):
        print("-", nome)


# ==========================================
# 9. SALVAR OS RESULTADOS
# ==========================================

tabela_resultados.to_csv(
    "resultados/comparacao_modelos.csv",
    index=False
)

coeficientes.to_csv(
    "resultados/comparacao_coeficientes.csv",
    index=False
)

print("\n==========================================")
print("RESUMO FINAL")
print("==========================================")

print("\nMelhor alpha Ridge:")
print(busca_ridge.best_params_)

print("\nMelhor alpha Lasso:")
print(busca_lasso.best_params_)

print("\nResultados e coeficientes salvos na pasta resultados.")

print("\nExecucao dos modelos de regressao concluida!")

"""GRAFICOS"""

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# 10. VISUALIZACAO DOS RESULTADOS
# ==========================================

print("\n==========================================")
print("SECAO 10 - VISUALIZACAO DOS RESULTADOS")
print("==========================================")

# Criar tabela completa de metricas

metricas = []

modelos = {
    "Regressao Linear": previsoes_linear,
    "Ridge": previsoes_ridge,
    "Lasso": previsoes_lasso
}

for nome, previsoes in modelos.items():

    mae = mean_absolute_error(y_test, previsoes)

    mse = mean_squared_error(y_test, previsoes)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, previsoes)

    metricas.append({
        "Modelo": nome,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

tabela_metricas = pd.DataFrame(metricas)

print("\nTabela de metricas:")
print(tabela_metricas.round(3).to_string(index=False))


# Gerar imagem da tabela

fig, ax = plt.subplots(figsize=(11, 3))

ax.axis("off")

tabela = ax.table(
    cellText=tabela_metricas.round(3).values,
    colLabels=tabela_metricas.columns,
    cellLoc="center",
    loc="center"
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.scale(1.2, 2)

plt.title("Metricas de avaliacao dos modelos", pad=20)

plt.tight_layout()

plt.savefig(
    "resultados/tabela_metricas.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# TABELA COMPARATIVA DOS MODELOS
# ==========================================

tabela_comparativa = tabela_metricas.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

tabela_comparativa.insert(
    0,
    "Posicao",
    range(1, len(tabela_comparativa) + 1)
)

print("\nTabela comparativa:")
print(tabela_comparativa.round(3).to_string(index=False))


# Gerar imagem

fig, ax = plt.subplots(figsize=(11, 3))

ax.axis("off")

tabela = ax.table(
    cellText=tabela_comparativa.round(3).values,
    colLabels=tabela_comparativa.columns,
    cellLoc="center",
    loc="center"
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.scale(1.2, 2)

plt.title(
    "Comparacao de desempenho dos modelos",
    pad=20
)

plt.tight_layout()

plt.savefig(
    "resultados/tabela_comparativa.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# GRAFICOS: VALORES REAIS X PREVISTOS
# ==========================================

for nome, previsoes in modelos.items():

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=y_test,
        y=previsoes,
        alpha=0.4
    )

    # Linha de previsao perfeita

    minimo = min(y_test.min(), previsoes.min())
    maximo = max(y_test.max(), previsoes.max())

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
        label="Previsao perfeita"
    )

    plt.title(f"Valores reais x previstos - {nome}")

    plt.xlabel("Valores reais")
    plt.ylabel("Valores previstos")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    nome_arquivo = nome.lower().replace(" ", "_")

    plt.savefig(
        f"resultados/real_previsto_{nome_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# ANALISE DA DISTRIBUICAO DOS RESIDUOS
# ==========================================

for nome, previsoes in modelos.items():

    residuos = y_test - previsoes

    nome_arquivo = nome.lower().replace(" ", "_")

    # --------------------------------------
    # GRAFICO 1 - HISTOGRAMA DOS RESIDUOS
    # --------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        residuos,
        kde=True,
        bins=30
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Residuo zero"
    )

    plt.title(f"Distribuicao dos residuos - {nome}")

    plt.xlabel("Residuos")
    plt.ylabel("Frequencia")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"resultados/residuos_histograma_{nome_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


    # --------------------------------------
    # GRAFICO 2 - RESIDUOS X PREVISTOS
    # --------------------------------------

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x=previsoes,
        y=residuos,
        alpha=0.4
    )

    plt.axhline(
        0,
        linestyle="--",
        label="Residuo zero"
    )

    plt.title(f"Residuos x valores previstos - {nome}")

    plt.xlabel("Valores previstos")
    plt.ylabel("Residuos")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"resultados/residuos_previstos_{nome_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# RESUMO ESTATISTICO DOS RESIDUOS
# ==========================================

resumo_residuos = []

for nome, previsoes in modelos.items():

    residuos = y_test - previsoes

    resumo_residuos.append({

        "Modelo": nome,

        "Media": residuos.mean(),

        "Desvio padrao": residuos.std(),

        "Minimo": residuos.min(),

        "Maximo": residuos.max(),

        "Mediana": residuos.median()

    })

tabela_residuos = pd.DataFrame(resumo_residuos)

print("\nResumo estatistico dos residuos:")

print(
    tabela_residuos.round(3).to_string(index=False)
)


# Salvar imagem da tabela

fig, ax = plt.subplots(figsize=(12, 3))

ax.axis("off")

tabela = ax.table(
    cellText=tabela_residuos.round(3).values,
    colLabels=tabela_residuos.columns,
    cellLoc="center",
    loc="center"
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 2)

plt.title("Resumo estatistico dos residuos", pad=20)

plt.tight_layout()

plt.savefig(
    "resultados/tabela_resumo_residuos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# 11. COMPARACAO FINAL DOS MODELOS
# ==========================================

print("\n==========================================")
print("COMPARACAO FINAL DOS MODELOS")
print("==========================================")

# Organizar os resultados pelo menor RMSE

comparacao_final = tabela_metricas.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print("\nDesempenho dos modelos no conjunto de teste:")
print(comparacao_final.round(3).to_string(index=False))


# Identificar o melhor modelo pelo RMSE

melhor_modelo = comparacao_final.iloc[0]

print("\n==========================================")
print("MELHOR MODELO IDENTIFICADO")
print("==========================================")

print("\nModelo:", melhor_modelo["Modelo"])

print("MAE:", round(melhor_modelo["MAE"], 3))

print("MSE:", round(melhor_modelo["MSE"], 3))

print("RMSE:", round(melhor_modelo["RMSE"], 3))

print("R2:", round(melhor_modelo["R2"], 3))


# Comparar os modelos graficamente

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=comparacao_final,
    x="Modelo",
    y="RMSE"
)

plt.title("Comparacao do RMSE dos modelos no conjunto de teste")

plt.xlabel("Modelos")
plt.ylabel("RMSE")

plt.tight_layout()

plt.savefig(
    "resultados/comparacao_final_rmse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# Gerar conclusao automatica

print("\n==========================================")
print("CONCLUSAO DA COMPARACAO")
print("==========================================")

print(
    f"O modelo {melhor_modelo['Modelo']} apresentou "
    f"o menor RMSE ({melhor_modelo['RMSE']:.3f}) "
    "no conjunto de teste."
)

print(
    f"Seu MAE foi de {melhor_modelo['MAE']:.3f}, "
    f"seu MSE foi de {melhor_modelo['MSE']:.3f} "
    f"e seu R2 foi de {melhor_modelo['R2']:.3f}."
)

print(
    "\nConsiderando o RMSE como criterio principal, "
    "este modelo apresentou o melhor desempenho "
    "entre os modelos avaliados."
)

print("\nComparacao final concluida!")