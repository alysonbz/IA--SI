from escalas import X_treino_final, X_teste_final, y_treino, y_teste
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Separando uma parte do treino para VALIDAÇÃO (o teste fica reservado)
X_train, X_val, y_train, y_val = train_test_split(X_treino_final, y_treino, test_size=0.25,
                                                  random_state=42, stratify=y_treino)

# Análise do KNN - alterando automaticamente o valor de k
train_accuracies = {}
val_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    val_accuracies[neighbor] = knn.score(X_val, y_val)

# Tabela de resultados
print("K | Acurácia de treinamento | Acurácia de validação")
for neighbor in neighbors:
    print(neighbor, "|", round(train_accuracies[neighbor], 4), "|", round(val_accuracies[neighbor], 4))

# Qual K teve a maior acurácia de validação?
melhor_k = 1
for neighbor in neighbors:
    if val_accuracies[neighbor] > val_accuracies[melhor_k]:
        melhor_k = neighbor
print("Melhor K na validação:", melhor_k, "->", val_accuracies[melhor_k])

# Modelo final: usa o K escolhido e avalia no TESTE uma única vez.
k_final = 5

knn = KNeighborsClassifier(n_neighbors=k_final)
knn.fit(X_treino_final, y_treino)
print("Acurácia no teste (K =", k_final, "):", knn.score(X_teste_final, y_teste))

# Plotando resultados
plt.figure(figsize=(8, 6))
plt.title("KNN: Variação do Número de Vizinhos")
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de Treinamento")
plt.plot(neighbors, val_accuracies.values(), label="Acurácia de Validação")
plt.legend()
plt.xlabel("Número de Vizinhos (K)")
plt.ylabel("Acurácia")
plt.show()

