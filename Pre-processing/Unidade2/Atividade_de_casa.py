from ucimlrepo import fetch_ucirepo
import math

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# metadata
print(iris.metadata)

# variable information
print(iris.variables)

mapa_classes = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}
lista = []

with open('../dataset/iris_data.csv', 'r') as f:
    for linha in f:
        linha = linha.strip()
        if not linha:
            continue
        valores = linha.split(',')
        amostra = [float(v) for v in valores[:4]] + [mapa_classes[valores[4]]]
        lista.append(amostra)


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


def dist_euclidiana(v1, v2):
            dim, soma = len(v1), 0
            for i in range(dim - 1):
                soma += math.pow(v1[i] - v2[i], 2)
            return math.sqrt(soma)

def dist_manhattan(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += abs(v1[i] - v2[i])
    return soma


def dist_chebyshev(v1, v2):
    dim = len(v1)
    maior_diferenca = 0
    for i in range(dim - 1):
        diferenca = abs(v1[i] - v2[i])
        if diferenca > maior_diferenca:
            maior_diferenca = diferenca
    return maior_diferenca


def dist_minkowski(v1, v2, p=3):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(abs(v1[i] - v2[i]), p)
    return math.pow(soma, 1/p)


def knn(treinamento, nova_amostra, K, funcao_distancia):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = funcao_distancia(treinamento[i], nova_amostra)  # <- usa o parâmetro, não fixo
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

acertos, K = 0, 3
for amostra in teste:
    classe = knn(treinamento, amostra, K, dist_euclidiana)
    if amostra[-1]==classe:
        acertos +=1
#print("Porcentagem de acertos:",100*acertos/len(teste))
distancias = {
    "Euclidiana": dist_euclidiana,
    "Manhattan": dist_manhattan,
    "Chebyshev": dist_chebyshev,
    "Minkowski": dist_minkowski,
}

K = 3
for nome, funcao in distancias.items():
    acertos = 0
    for amostra in teste:
        classe = knn(treinamento, amostra, K, funcao)
        if amostra[-1] == classe:
            acertos += 1
    print(f"Distância {nome}: {100 * acertos / len(teste):.2f}% de acertos")