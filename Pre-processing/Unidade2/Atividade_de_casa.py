import math
import pandas as pd

# 1. Carregamento dos dados (Carrega o Iris dataset diretamente da fonte oficial)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)

# Mapeia as classes de texto para os valores numéricos (1.0, 2.0, 3.0)
mapeamento = {
    'Iris-setosa': 1.0,
    'Iris-versicolor': 2.0,
    'Iris-virginica': 3.0,
}
df[4] = df[4].map(mapeamento)
lista = df.values.tolist()


# 2. Contagem das classes
def countclasses(lista):
    setosa, versicolor, virginica = 0, 0, 0
    for i in range(len(lista)):
        if float(lista[i][4]) == 1.0:
            setosa += 1
        elif float(lista[i][4]) == 2.0:
            versicolor += 1
        elif float(lista[i][4]) == 3.0:
            virginica += 1
    return [setosa, versicolor, virginica]


# 3. Divisão entre Treino (60%) e Teste (40%)
p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = (
    int(p * setosa),
    int(p * versicolor),
    int(p * virginica),
)

total1, total2, total3 = 0, 0, 0
for lis in lista:
    classe_val = float(lis[-1])
    if classe_val == 1.0 and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif classe_val == 2.0 and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif classe_val == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)


# 4. Cálculo das 4 Distâncias Solicitadas no Slide
def calcular_distancia(v1, v2, tipo_distancia='euclidiana', p_minkowski=3):
    dim = len(v1) - 1  # Ignora a coluna de classe
    diferencas = [abs(float(v1[i]) - float(v2[i])) for i in range(dim)]

    if tipo_distancia == 'euclidiana':
        return math.sqrt(sum(d**2 for d in diferencas))

    elif tipo_distancia == 'manhattan':
        return sum(diferencas)

    elif tipo_distancia == 'minkowski':
        return sum(d**p_minkowski for d in diferencas) ** (1 / p_minkowski)

    elif tipo_distancia == 'chebyshev':
        return max(diferencas)


# 5. Algoritmo KNN Genérico
def knn(treinamento, nova_amostra, K, tipo_distancia='euclidiana'):
    dists = {}
    for i in range(len(treinamento)):
        dists[i] = calcular_distancia(
            treinamento[i], nova_amostra, tipo_distancia
        )

    # Seleciona os K vizinhos mais próximos
    k_vizinhos = sorted(dists, key=dists.get)[:K]

    # Contagem de frequência das classes nos K vizinhos
    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        classe_v = float(treinamento[indice][-1])
        if classe_v == 1.0:
            qtd_setosa += 1
        elif classe_v == 2.0:
            qtd_versicolor += 1
        elif classe_v == 3.0:
            qtd_virginica += 1

    frequencias = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return float(frequencias.index(max(frequencias)) + 1.0)


# 6. Comparação dos Resultados das 4 Distâncias (Requisito da Tarefa)
distancias = ['euclidiana', 'manhattan', 'minkowski', 'chebyshev']
K = 1

print("=== COMPARATIVO DE ACURÁCIA DO KNN MANUAL ===")
for dist in distancias:
    acertos = 0
    for amostra in teste:
        classe_predita = knn(treinamento, amostra, K, tipo_distancia=dist)
        if float(amostra[-1]) == classe_predita:
            acertos += 1

    porcentagem = (acertos / len(teste)) * 100
    print(f"Distância {dist.capitalize():<12}: {porcentagem:.2f}% de acertos")