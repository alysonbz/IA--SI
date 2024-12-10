import math

# Dicionário para mapear classes de texto para números
classe_para_numero = {
    'Iris-setosa': 1.0,
    'Iris-versicolor': 2.0,
    'Iris-virginica': 3.0
}

lista = []
with open('iris_data.csv', 'r') as f:
    for linha in f.readlines():
        a = linha.strip().split(',')  # Remove espaços e divide por vírgula
        # Converter todas as colunas exceto a última para float
        a[:-1] = [float(x) for x in a[:-1]]
        # Mapear a classe (última coluna) para número
        a[-1] = classe_para_numero[a[-1]]
        lista.append(a)

# Exibir algumas linhas para verificar
print(lista[:20])


# Contar classes
def countclasses(lista):
    setosa, versicolor, virginica = 0, 0, 0
    for i in range(len(lista)):
        if lista[i][-1] == 1.0:
            setosa += 1
        elif lista[i][-1] == 2.0:
            versicolor += 1
        elif lista[i][-1] == 3.0:
            virginica += 1
    return [setosa, versicolor, virginica]

# Separar treinamento e teste
p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
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

# Função para calcular a distância chebyshev
def dist_chebyshev(v1, v2):
    dim, max_diff = len(v1), 0
    for i in range(dim - 1):
        max_diff = max(max_diff, abs(v1[i] - v2[i]))
    return max_diff

# Função KNN
def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_chebyshev(treinamento[i], nova_amostra)
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

# Avaliação do modelo
acertos, K = 0, 3  # Teste com K = 3
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1

print("Porcentagem de acertos:", 100 * acertos / len(teste))

