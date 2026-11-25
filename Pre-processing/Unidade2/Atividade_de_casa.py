import math

# 1. Mapeamento e leitura do dataset
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

# 2. Contagem de amostras por classe
def countclasses(lista):
    setosa, versicolor, virginica = 0, 0, 0
    for amostra in lista:
        if amostra[4] == 1.0:
            setosa += 1
        elif amostra[4] == 2.0:
            versicolor += 1
        elif amostra[4] == 3.0:
            virginica += 1
    return [setosa, versicolor, virginica]

# 3. Divisão Estratificada (60% para treino)
p = 0.6
setosa, versicolor, virginica = countclasses(lista)
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)

treinamento, teste = [], []
total1, total2, total3 = 0, 0, 0

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

# 4. Cálculo da Distância Euclidiana
def dist_euclidiana(v1, v2):
    soma = 0.0
    for i in range(len(v1) - 1):
        soma += (v1[i] - v2[i]) ** 2
    return math.sqrt(soma)

# 5. Algoritmo KNN
def knn(treinamento, nova_amostra, K):
    dists = {}
    for i in range(len(treinamento)):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    # Ordena os índices das menores distâncias
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
    return float(a.index(max(a)) + 1)

# 6. Avaliação de Desempenho
acertos = 0
K = 3

for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1

acuracia = (acertos / len(teste)) * 100
print(f"Porcentagem de acertos (K={K}): {acuracia:.2f}%")

import math
from collections import Counter
########################################################################distância euclidiana############################################################################
# 1. Mapeamento das classes, converter pra float tmb
map_class= {
    'Iris-setosa': 1.0,
    'Iris-versicolor': 2.0,
    'Iris-virginica': 3.0}
def load_iris_dataset():
    dataset = []
    with open('../dataset/iris_data.csv', 'r') as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            valores = linha.split(',')
            amostra = [float(v) for v in valores[:4]] + [map_class[valores[4]]]
            dataset.append(amostra)
    return dataset

# 3. Divisão Estratificada
def divisao_estratificada(dataset, p=0.6):
    classes = {}
    for linha in dataset:
        rotulo = linha[-1]
        classes[rotulo] = classes.get(rotulo, 0) + 1
    limites = {rotulo: int(qtd * p) for rotulo, qtd in classes.items()}
    contagem = {rotulo: 0 for rotulo in classes}

    treinamento, teste = [], []
    for linha in dataset:
        rotulo = linha[-1]
        if contagem[rotulo] < limites[rotulo]:
            treinamento.append(linha)
            contagem[rotulo] += 1
        else:
            teste.append(linha)

    return treinamento, teste

# 4. Distância Euclidiana
def dist_euclidiana(v1, v2):
    soma = 0.0
    for i in range(len(v1) - 1):
        soma += (v1[i] - v2[i]) ** 2
    return math.sqrt(soma)

# 5. KNN
def knn_prever(treinamento, nova_amostra, k):
    distancias = []
    for linha in treinamento:
        d = dist_euclidiana(linha, nova_amostra)
        distancias.append((linha[-1], d))

    distancias.sort(key=lambda x: x[1])
    k_vizinhos = [rotulo for rotulo, _ in distancias[:k]]
    votos = Counter(k_vizinhos)
    return votos.most_common(1)[0][0]

dataset = load_iris_dataset()
treinamento, teste = divisao_estratificada(dataset, p=0.6)

K = 3
acertos = 0

for amostra in teste:
    previsao = knn_prever(treinamento, amostra, K)
    if amostra[-1] == previsao:
        acertos += 1

acuracia = (acertos / len(teste)) * 100
print(f"Porcentagem de acertos (K={K}): {acuracia:.2f}%")

########################################mahattan##########################
import math

# 1. Mapeamento e leitura do dataset
mapa_classes = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}
listam = []

with open('../dataset/iris_data.csv', 'r') as f:
    for linha in f:
        linha = linha.strip()
        if not linha:
            continue
        valores = linha.split(',')
        amostra = [float(v) for v in valores[:4]] + [mapa_classes[valores[4]]]
        listam.append(amostra)

# 2. Contagem de amostras por classe
treinamento, teste = divisao_estratificada(lista)
# 4. Cálculo da Distância Manhattan
def dist_manhattan(v1, v2):
    soma = 0.0
    for i in range(len(v1) - 1):
        soma += abs(v1[i] - v2[i])  # Soma dos valores absolutos (módulo)
    return soma                     # Não usa raiz quadrada

# 5. Algoritmo KNN ajustado para Manhattan
def knn(treinamento, nova_amostra, K):
    dists = {}
    for i in range(len(treinamento)):
        d = dist_manhattan(treinamento[i], nova_amostra)
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
    return float(a.index(max(a)) + 1)

# 6. Avaliação de Desempenho
acertos = 0
K = 3

for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1

acuracia = (acertos / len(teste)) * 100
print(f"Porcentagem de acertos com Manhattan (K={K}): {acuracia:.2f}%")

#######kowaski###
#mapa de classes ou botar um contador para ver quanto cada classe tem
import math

mapa_classes = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}
lista = []

with open('../dataset/iris_data.csv', 'r') as arquivo:
    for linha in arquivo:
        linha = linha.strip()
        if not linha:
            continue
        valores = linha.split(",")
        amostra = [float(v) for v in valores[:4]] + [mapa_classes[valores[4]]]
        lista.append(amostra)

# ja tinha usado a mesma função antes,divisao_estratificada()
treinamento, teste = divisao_estratificada(lista)
def distancia_minkowki(v1, v2, p=3):
    soma=0.0
    for i in range(len(v1)-1):
        soma += abs(v1[i] - v2[i]) ** p
    return soma ** (1/p)
def knn(treinamento, nova_amostra, K, p_minkowski=3):
    distancia = {}
    for i in range(len(treinamento)):
        d = distancia_minkowki(treinamento[i], nova_amostra, p=p_minkowski)
        distancia[i] = d
    k_vizinhos = sorted(distancia, key=distancia.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return float(a.index(max(a)) + 1)
acertos = 0
K = 3
P_ORDEM = 3

for amostra in teste:
    classe = knn(treinamento, amostra, K, p_minkowski=P_ORDEM)
    if amostra[-1] == classe:
        acertos += 1

acuracia = (acertos / len(teste)) * 100
print(f"Acurácia com Distância de Minkowski (K={K}, p={P_ORDEM}): {acuracia:.2f}%")

########################cherbyshev#####################################
#considera apenas a maior diferença individual entre as coordenadas de dois pontos, ignorando a soma das demais#
# 1. Função de Distância de Chebyshev
def dist_chebyshev(v1, v2):
    diferencas = []
    for i in range(len(v1) - 1):
        diferenca_abs = abs(v1[i] - v2[i])
        diferencas.append(diferenca_abs)
    return max(diferencas)


# 2. Algoritmo KNN usando Chebyshev
def knn(treinamento, nova_amostra, K):
    distancia = {}
    for i in range(len(treinamento)):
        d = dist_chebyshev(treinamento[i], nova_amostra)
        distancia[i] = d
    k_vizinhos = sorted(distancia, key=distancia.get)[:K]
    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return float(a.index(max(a)) + 1)
# 3. Bloco de Cálculo de Acertos e Acurácia
acertos = 0
K = 3
for amostra in teste:
    classe_prevista = knn(treinamento, amostra, K)
    classe_real = amostra[-1]
    if classe_real == classe_prevista:
        acertos += 1
total_testes = len(teste)
acuracia = (acertos / total_testes) * 100
print(f"Acurácia com Distância de Chebyshev (K={K}): {acuracia:.2f}%")