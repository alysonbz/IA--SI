
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ==========================================
# 1. CONFIGURACOES INICIAIS
# ==========================================

# Localizar o dataset a partir da pasta do projeto

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_DATASET = PASTA_PROJETO / "dataset" / "hour.csv"

df = pd.read_csv(CAMINHO_DATASET)

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
