import random
def criar_matriz(tamanho):
"""Cria uma matriz quadrada aleatória de tamanho n x n."""
return [[random.randint(1, 10) for _ in range(tamanho)] for _ in range(tamanho)]
def multiplicar_matrizes(A, B):
"""Multiplica duas matrizes quadradas A e B."""
n = len(A)
return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
# Entrada do usuário
tamanho = int(input("Digite o tamanho das matrizes: "))
# Criação das matrizes
matriz_a = criar_matriz(tamanho)
matriz_b = criar_matriz(tamanho)
# Multiplicação das matrizes
resultado = multiplicar_matrizes(matriz_a, matriz_b)
# Impressão das matrizes