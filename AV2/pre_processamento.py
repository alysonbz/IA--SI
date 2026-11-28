
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 1. Criar a pasta para salvar os resultados
Path("dataset").mkdir(exist_ok=True)

# 2. Carregar o dataset por hora
df = pd.read_csv("dataset/hour.csv")

print("DADOS ORIGINAIS:")
print("Quantidade de linhas:", len(df))

# 3. Remover registros duplicados
df = df.drop_duplicates()

# 4. Remover linhas com valores ausentes
df = df.dropna()

# 5. Remover registros inconsistentes
df = df[df["cnt"] == df["casual"] + df["registered"]]

print("\nDADOS APÓS A LIMPEZA:")
print("Quantidade de linhas:", len(df))
print("Valores ausentes:", df.isnull().sum().sum())
print("Duplicados:", df.duplicated().sum())

# 6. Selecionar as características para o K-Means
colunas_cluster = [
    "season",
    "mnth",
    "hr",
    "weekday",
    "weathersit",
    "yr",
    "holiday",
    "workingday",
    "temp",
    "hum",
    "windspeed"
]

X = df[colunas_cluster].copy()

# 7. Transformar variáveis categóricas em números
colunas_categoricas = [
    "season",
    "mnth",
    "hr",
    "weekday",
    "weathersit"
]

X = pd.get_dummies(
    X,
    columns=colunas_categoricas,
    dtype=int
)

# 8. Preservar os dados originais para interpretar os clusters
colunas_interpretacao = [
    "season",
    "mnth",
    "hr",
    "weekday",
    "weathersit",
    "yr",
    "holiday",
    "workingday",
    "temp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt"
]

dados_antes_escala = df[colunas_interpretacao].copy()

dados_antes_escala.to_csv(
    "dataset/dados_antes_escala.csv",
    index=False
)

# 9. Padronizar as características utilizadas no K-Means
scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(X)

dados_padronizados = pd.DataFrame(
    dados_padronizados,
    columns=X.columns
)

# 10. Salvar os dados padronizados
dados_padronizados.to_csv(
    "dataset/dados_padronizados.csv",
    index=False
)


print("Quantidade de características após a transformação:", X.shape[1])

