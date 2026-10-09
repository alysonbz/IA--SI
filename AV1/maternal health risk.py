import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score


# Carregar o dataset
df = pd.read_csv("Maternal Health Risk Data Set.csv")

# Dimensões
print("Dimensões do dataset:")
print(df.shape)

# Tipos das variáveis
print("\nTipos das variáveis:")
print(df.dtypes)

# Colunas
print("\nColunas:")
print(df.columns)

# Variável de classe
print("\nVariável de classe:")
print(df["RiskLevel"].value_counts())


# Verificar valores ausentes
print("\nValores ausentes:")
print(df.isnull().sum())

# Verificar registros duplicados
print("\nQuantidade de registros duplicados:")
print(df.duplicated().sum())



# Remover apenas duplicatas exatas para análise
df_sem_duplicatas = df.drop_duplicates()
print("\nDimensões após remover duplicatas:")
print(df_sem_duplicatas.shape)

print("\nDistribuição das classes após remover duplicatas:")
print(df_sem_duplicatas["RiskLevel"].value_counts())

# Verificar características iguais com diferentes classes
features = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]

grupos_conflitantes = (
    df_sem_duplicatas
    .groupby(features)["RiskLevel"]
    .nunique()
)

conflitos = grupos_conflitantes[grupos_conflitantes > 1]

print("\nQuantidade de combinações de características com classes diferentes:")
print(len(conflitos))

# Identificar registros pertencentes a grupos conflitantes
df_preparado = df_sem_duplicatas[
    ~df_sem_duplicatas.set_index(features).index.isin(conflitos.index)
].copy()

print("\nDimensões do dataset após remover conflitos:")
print(df_preparado.shape)

print("\nRegistros removidos por conflitos:")
print(len(df_sem_duplicatas) - len(df_preparado))

print("\nDistribuição final das classes:")
print(df_preparado["RiskLevel"].value_counts())

print("\nResumo da preparação dos dados:")
print("Dataset original:", df.shape)
print("Após remover duplicatas:", df_sem_duplicatas.shape)
print("Duplicatas removidas:", len(df) - len(df_sem_duplicatas))
print("Após remover conflitos:", df_preparado.shape)
print("Registros removidos por conflitos:", len(df_sem_duplicatas) - len(df_preparado))

print("\nEstatísticas descritivas das variáveis numéricas:")
print(df_preparado[features].describe())



# Distribuição das classes
distribuicao = df_preparado["RiskLevel"].value_counts()

print("\nDistribuição final das classes:")
print(distribuicao)

plt.figure(figsize=(7, 5))
distribuicao.plot(kind="bar")

plt.title("Distribuição das classes de risco")
plt.xlabel("Classe de risco")
plt.ylabel("Quantidade de registros")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print("\nAvaliação das escalas das variáveis:")

avaliacao_escala = pd.DataFrame({
    "Mínimo": df_preparado[features].min(),
    "Máximo": df_preparado[features].max(),
    "Amplitude": df_preparado[features].max() - df_preparado[features].min()
})

print(avaliacao_escala)



# Separar variáveis preditoras e variável de classe
X = df_preparado[features]
y = df_preparado["RiskLevel"]

# Separar treinamento (60%) e restante (40%)
X_treino, X_temp, y_treino, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Separar validação (20%) e teste (20%)
X_validacao, X_teste, y_validacao, y_teste = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDivisão dos dados:")
print("Treinamento:", X_treino.shape)
print("Validação:", X_validacao.shape)
print("Teste:", X_teste.shape)

# ==========================================================
# QUESTÃO 2 - COMPARAÇÃO DAS TÉCNICAS DE NORMALIZAÇÃO
# ==========================================================

resultados_normalizacao = []

# ----------------------------------------------------------
# 1. Sem normalização
# ----------------------------------------------------------

modelo = KNeighborsClassifier(n_neighbors=5)

modelo.fit(X_treino, y_treino)

previsoes_treino = modelo.predict(X_treino)
previsoes_validacao = modelo.predict(X_validacao)

acuracia_treino = accuracy_score(y_treino, previsoes_treino)
acuracia_validacao = accuracy_score(y_validacao, previsoes_validacao)

resultados_normalizacao.append({
    "Técnica": "Sem normalização",
    "Acurácia Treino": acuracia_treino,
    "Acurácia Validação": acuracia_validacao
})


# ----------------------------------------------------------
# 2. StandardScaler
# ----------------------------------------------------------

scaler = StandardScaler()

X_treino_scaled = scaler.fit_transform(X_treino)
X_validacao_scaled = scaler.transform(X_validacao)

modelo = KNeighborsClassifier(n_neighbors=5)

modelo.fit(X_treino_scaled, y_treino)

previsoes_treino = modelo.predict(X_treino_scaled)
previsoes_validacao = modelo.predict(X_validacao_scaled)

acuracia_treino = accuracy_score(y_treino, previsoes_treino)
acuracia_validacao = accuracy_score(y_validacao, previsoes_validacao)

resultados_normalizacao.append({
    "Técnica": "StandardScaler",
    "Acurácia Treino": acuracia_treino,
    "Acurácia Validação": acuracia_validacao
})


# ----------------------------------------------------------
# 3. MinMaxScaler
# ----------------------------------------------------------

scaler = MinMaxScaler()

X_treino_scaled = scaler.fit_transform(X_treino)
X_validacao_scaled = scaler.transform(X_validacao)

modelo = KNeighborsClassifier(n_neighbors=5)

modelo.fit(X_treino_scaled, y_treino)

previsoes_treino = modelo.predict(X_treino_scaled)
previsoes_validacao = modelo.predict(X_validacao_scaled)

acuracia_treino = accuracy_score(y_treino, previsoes_treino)
acuracia_validacao = accuracy_score(y_validacao, previsoes_validacao)

resultados_normalizacao.append({
    "Técnica": "MinMaxScaler",
    "Acurácia Treino": acuracia_treino,
    "Acurácia Validação": acuracia_validacao
})


# ----------------------------------------------------------
# 4. RobustScaler
# ----------------------------------------------------------

scaler = RobustScaler()

X_treino_scaled = scaler.fit_transform(X_treino)
X_validacao_scaled = scaler.transform(X_validacao)

modelo = KNeighborsClassifier(n_neighbors=5)

modelo.fit(X_treino_scaled, y_treino)

previsoes_treino = modelo.predict(X_treino_scaled)
previsoes_validacao = modelo.predict(X_validacao_scaled)

acuracia_treino = accuracy_score(y_treino, previsoes_treino)
acuracia_validacao = accuracy_score(y_validacao, previsoes_validacao)

resultados_normalizacao.append({
    "Técnica": "RobustScaler",
    "Acurácia Treino": acuracia_treino,
    "Acurácia Validação": acuracia_validacao
})


# ----------------------------------------------------------
# Mostrar resultados
# ----------------------------------------------------------

tabela_resultados = pd.DataFrame(resultados_normalizacao)

print("\nComparação das técnicas de normalização:")
print(tabela_resultados.to_string(index=False))

# Gráfico de comparação das técnicas de normalização

plt.figure(figsize=(9, 5))

plt.bar(
    tabela_resultados["Técnica"],
    tabela_resultados["Acurácia Validação"]
)

plt.title("Comparação das técnicas de normalização")
plt.xlabel("Técnica")
plt.ylabel("Acurácia de validação")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# ==========================================================
# QUESTÃO 3 - INVESTIGAÇÃO DO MELHOR VALOR DE K
# ==========================================================

# Padronizar os dados usando StandardScaler
scaler = StandardScaler()

X_treino_scaled = scaler.fit_transform(X_treino)
X_validacao_scaled = scaler.transform(X_validacao)

# Valores de K que serão avaliados
valores_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

resultados_k = []

for k in valores_k:

    modelo = KNeighborsClassifier(n_neighbors=k)

    modelo.fit(X_treino_scaled, y_treino)

    previsoes_treino = modelo.predict(X_treino_scaled)
    previsoes_validacao = modelo.predict(X_validacao_scaled)

    acuracia_treino = accuracy_score(
        y_treino,
        previsoes_treino
    )

    acuracia_validacao = accuracy_score(
        y_validacao,
        previsoes_validacao
    )

    resultados_k.append({
        "K": k,
        "Acurácia Treino": acuracia_treino,
        "Acurácia Validação": acuracia_validacao
    })


# Criar tabela com os resultados
tabela_k = pd.DataFrame(resultados_k)

print("\nResultados para diferentes valores de K:")
print(tabela_k.to_string(index=False))

# ==========================================================
# GRÁFICO - K x ACURÁCIA DE TREINAMENTO E VALIDAÇÃO
# ==========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

plt.plot(
    tabela_k["K"],
    tabela_k["Acurácia Treino"] * 100,
    marker="o",
    label="Acurácia de treinamento"
)

plt.plot(
    tabela_k["K"],
    tabela_k["Acurácia Validação"] * 100,
    marker="o",
    label="Acurácia de validação"
)

plt.title("Acurácia de treinamento e validação para diferentes valores de K")
plt.xlabel("Valor de K")
plt.ylabel("Acurácia (%)")

plt.xticks(tabela_k["K"])
plt.ylim(50, 105)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# AVALIAÇÃO FINAL DO MODELO
# ==========================================================

# Juntar os dados de treinamento e validação
X_treino_final = pd.concat([X_treino, X_validacao])
y_treino_final = pd.concat([y_treino, y_validacao])

# Padronizar os dados usando StandardScaler
scaler_final = StandardScaler()

X_treino_final_scaled = scaler_final.fit_transform(X_treino_final)
X_teste_final_scaled = scaler_final.transform(X_teste)

# Criar o modelo final
modelo_final = KNeighborsClassifier(n_neighbors=5)

# Treinar o modelo final
modelo_final.fit(X_treino_final_scaled, y_treino_final)

# Fazer previsões no conjunto de teste
previsoes_teste = modelo_final.predict(X_teste_final_scaled)

# Calcular acurácia final
acuracia_teste = accuracy_score(y_teste, previsoes_teste)

print("\nAvaliação final do modelo:")
print("Normalização: StandardScaler")
print("K escolhido: 5")
print("Dados utilizados no treinamento final:",
      X_treino_final.shape)
print("Dados utilizados no teste:",
      X_teste.shape)
print("Acurácia no conjunto de teste:",
      f"{acuracia_teste * 100:.2f}%")