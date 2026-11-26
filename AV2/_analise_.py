from _dados_ import *

# Dimensão do dataset
print("Número de amostras:", X.shape[0])
print("Número de características:", X.shape[1])
print("Número de variáveis-alvo:", y.shape[1])
print("Número total de colunas:", X.shape[1] + y.shape[1])

print("\nTipos de dados")
print(energy.dtypes)

# Estatísticas
var_num = energy[["X1", "X2", "X3", "X4", "X5", "X7"]]

print("\nEstatísticas descritivas")
estatisticas = var_num.describe().T[["mean", "50%", "std", "min", "max"]]
estatisticas.columns = ["Média", "Mediana", "Desvio-Padrão", "Mínimo", "Máximo"]
print(estatisticas)
