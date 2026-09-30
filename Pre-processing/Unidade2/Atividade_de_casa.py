from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# metadata
print(iris.metadata)

# variable information
print(iris.variables)

mapa_classe = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}
lista = []
with open('../dataset/iris_data.csv', 'r') as f:
    for i in range(len(X)):
        atributos = list(X.iloc[i])              # os 4 números daquela linha
        classe_texto = y.iloc[i, 0]               # o nome da classe (texto) daquela linha
        linha = atributos + [mapa_classe[classe_texto]]
        lista.append(linha)

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if lista[i][4] == 1.0:
            setosa += 1
        if lista[i][4] == 2.0:
            versicolor += 1
        if lista[i][4] == 3.0:
            virginica += 1

    return [setosa, versicolor, virginica]

p=0.6
setosa,versicolor, virginica = countclasses(lista)
treinamento, teste= [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1 =0
total2 =0
total3 =0
for lis in lista:
    if lis[-1]==1.0 and total1< max_setosa:
        treinamento.append(lis)
        total1 +=1
    elif lis[-1]==2.0 and total2<max_versicolor:
        treinamento.append(lis)
        total2 +=1
    elif lis[-1]==3.0 and total3<max_virginica:
        treinamento.append(lis)
        total3 +=1
    else:
        teste.append(lis)

import math

def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)

def dist_manhattan(v1, v2):
    dim, soma_abs = len(v1), 0
    for i in range(dim - 1):
        soma_abs += abs(v1[i] - v2[i])
    return soma_abs

def dist_chebyshev(v1, v2):
    dim, max_diff = len(v1), 0
    for i in range(dim - 1):
        diff = abs(v1[i] - v2[i])
        if diff > max_diff:
            max_diff = diff
    return max_diff

def dist_minkowski(v1, v2, p):
    dim, soma_pot = len(v1), 0
    for i in range(dim - 1):
        soma_pot += math.pow(abs(v1[i] - v2[i]), p)
    return math.pow(soma_pot, 1/p)

def knn_generalized(treinamento, nova_amostra, K, dist_func, p=None):
    dists = {}
    for i in range(len(treinamento)):
        if p is not None and dist_func.__name__ == 'dist_minkowski':
            d = dist_func(treinamento[i], nova_amostra, p)
        else:
            d = dist_func(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0

K = 3

for nome, funcao, extra in [
    ("Euclidiana", dist_euclidiana, {}),
    ("Manhattan", dist_manhattan, {}),
    ("Chebyshev", dist_chebyshev, {}),
    ("Minkowski, p=3", dist_minkowski, {"p": 3}),
]:
    acertos = 0
    for amostra in teste:
        classe = knn_generalized(treinamento, amostra, K, funcao, **extra)
        if amostra[-1] == classe:
            acertos += 1
    print(f"Porcentagem de acertos ({nome}): {100 * acertos / len(teste):.2f}%")