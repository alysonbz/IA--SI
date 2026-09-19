import math

lista = []

with open("../dataset/iris_data.csv", "r") as f:
    for linha in f.readlines():
        linha = linha.strip()

        if linha == "":
            continue

        dados = linha.split(",")

        try:
            dados[:4] = [float(x) for x in dados[:4]]
        except ValueError:
            continue

        if dados[4] == "Iris-setosa":
            dados[4] = 1.0
        elif dados[4] == "Iris-versicolor":
            dados[4] = 2.0
        elif dados[4] == "Iris-virginica":
            dados[4] = 3.0
        else:
            continue

        lista.append(dados)

print("Quantidade de amostras:", len(lista))


def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0

    for i in range(len(lista)):
        if lista[i][4] == 1.0:
            setosa += 1
        elif lista[i][4] == 2.0:
            versicolor += 1
        elif lista[i][4] == 3.0:
            virginica += 1

    return [setosa, versicolor, virginica]


setosa, versicolor, virginica = countclasses(lista)

print("Iris-setosa:", setosa)
print("Iris-versicolor:", versicolor)
print("Iris-virginica:", virginica)


p = 0.6

treinamento = []
teste = []

max_setosa = int(p * setosa)
max_versicolor = int(p * versicolor)
max_virginica = int(p * virginica)

total1 = 0
total2 = 0
total3 = 0

for lis in lista:
    if lis[-1] == 1.0 and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[-1] == 2.0 and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[-1] == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)

print("Dados de treinamento:", len(treinamento))
print("Dados de teste:", len(teste))


def dist_euclidiana(v1, v2):
    dim = len(v1)
    soma = 0

    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)

    return math.sqrt(soma)


def knn(treinamento, nova_amostra, K):
    dists = {}
    len_treino = len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa = 0
    qtd_versicolor = 0
    qtd_virginica = 0

    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    quantidade = [qtd_setosa, qtd_versicolor, qtd_virginica]

    return quantidade.index(max(quantidade)) + 1.0


acertos = 0
K = 1

for amostra in teste:
    classe = knn(treinamento, amostra, K)

    if amostra[-1] == classe:
        acertos += 1

print("Porcentagem de acertos:", 100 * acertos / len(teste))