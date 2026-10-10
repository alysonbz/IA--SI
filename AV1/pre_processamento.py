from dados import *

# Verificando o dataset
print("Informações do dataset")
print(heart.info())

# Verificando a dimensão do dataset
print("\nDimensão do dataset:",heart.shape)

# Variável de Classe (Target)
print("\nVariável Classe:", y.columns[0])
