import math

lista = []
with open('C:\\Users\\vitor\\Downloads\\IA.BLACK\\IA--SI\\Pre-processing\\dataset\\iris_data.csv','r')as f:
    for linha in f.readlines():
        a = linha.strip().split(',')
        lista.append(a)

def countclasses(lista):
    setosa,versicolor,virginica = 0,0,0
    for i in range(len(lista)):
        if lista[i][4] =='Iris-setosa':
            setosa+=1
        elif lista[i][4] =='Iris-versicolor':
            versicolor += 1
        elif lista[i][4] =='Iris-virginica':
            virginica += 1
    return [setosa, versicolor, virginica]

p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
total1, total2, total3 = 0, 0, 0

for lis in lista:
    if lis[-1] == 'Iris-setosa' and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[-1] == 'Iris-versicolor' and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[-1] == 'Iris-virginica' and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)

def dist_euclidiana(v1, v2):
    return math.sqrt(sum((float(v1[i]) - float(v2[i])) ** 2 for i in range(len(v1) - 1)))

def knn(treinamento, nova_amostra, K):
    dists = {}
    for i, amostra in enumerate(treinamento):
        d = dist_euclidiana(amostra, nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 'Iris-setosa':

            qtd_setosa += 1
        elif treinamento[indice][-1] == 'Iris-versicolor':
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    return [qtd_setosa, qtd_versicolor, qtd_virginica].index(max([qtd_setosa, qtd_versicolor, qtd_virginica])) + 1.0

acertos, K = 0, 3

for amostra in teste:
    classe_predita = knn(treinamento, amostra, K)
    if str(amostra[-1]) == str(classe_predita):  # Comparação correta entre strings
        acertos += 1

print("Porcentagem de acertos:", 100 * acertos / len(teste))
