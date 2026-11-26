from _dados_ import *

print("Verificando elementos nulos:")
print(energy.isna().sum())

print("\nQuantidade de valores duplicados:")
print(energy.duplicated().sum())

print("\nValores encontrados em X6:")
print(sorted(energy["X6"].unique()))

print("\nValores encontrados em X8:")
print(sorted(energy["X8"].unique()))

# Selecionar as variáveis numéricas
variaveis_numericas = ["X1", "X2", "X3", "X4", "X5", "X7", "Y1", "Y2"]


# Calcular os quartis
Q1 = energy[variaveis_numericas].quantile(0.25)
Q3 = energy[variaveis_numericas].quantile(0.75)

# Calcular o intervalo interquartil
IQR = Q3 - Q1

# Calcular os limites
limite_inferior = Q1 - (1.5 * IQR)
limite_superior = Q3 + (1.5 * IQR)

# Identificar a quantidade de outliers
outliers = ((energy[variaveis_numericas] < limite_inferior) |
            (energy[variaveis_numericas] > limite_superior)).sum()

print("\nQuantidade de outliers por variável:")
print(outliers)

