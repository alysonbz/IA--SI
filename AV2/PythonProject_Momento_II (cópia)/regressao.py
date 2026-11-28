
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. CARREGAMENTO DO DATASET
# ==========================================

df = pd.read_csv("dataset/hour.csv")

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
