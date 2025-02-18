import math
def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0

    for i in range(len(lista)):
        if len(lista[i]) >= 5:
            if lista[i][4] == 1.0:
                setosa += 1
            if lista[i][4] == 2.0:
                versicolor += 1
            if lista[i][4] == 3.0:
                virginica += 1

    return [setosa, versicolor, virginica]

def is_number(s):
    try:
        float(s)  # Tenta converter a string para um número
        return True
    except ValueError:
        return False
def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
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

lista=[]
with open('iris_data.csv', 'r') as f:
    for linha in f.readlines():
        a=linha.replace('\n','').split(',')
        lista.append(a)

for i in range(len(lista)):
    for j in range(len(lista[i])):
        cond = is_number(lista[i][j])
        if cond:
            number = float(lista[i][j])
            lista[i][j] = number
for i in lista:
    if len(i) >= 5:
        if isinstance(i[4], str):  # Garantir que está lidando com strings
            if i[4] == 'Iris-setosa':
                i[4] = 1.0
            elif i[4] == 'Iris-versicolor':
                i[4] = 2.0
            elif i[4] == 'Iris-virginica':
                i[4] = 3.0
for i in lista:
    if len(i) >= 5:
        print(type(i[0]), type(i[1]), type(i[2]), type(i[3]), type(i[4]))


p=0.6
setosa,versicolor, virginica = countclasses(lista)
treinamento, teste= [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1, total2, total3 = 0, 0, 0


max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1, total2, total3 = 0, 0, 0

for i in lista:
    if len(i) >= 5:
        if i[4] == 1.0 and total1 < max_setosa:
            treinamento.append(i)
            total1 += 1
        elif i[4] == 2.0 and total2 < max_versicolor:
            treinamento.append(i)
            total2 += 1
        elif i[4] == 3.0 and total3 < max_virginica:
            treinamento.append(i)
            total3 += 1
        else:
            teste.append(i)




def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print(len(teste))
print("Porcentagem de acertos:",100*acertos/len(teste))