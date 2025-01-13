import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def knn(X_train, y_train, X_test, k, metric):
    """
    Implementação manual do KNN com diferentes métricas.

    Parâmetros:
        X_train: Dados de treino (features)
        y_train: Rótulos de treino
        X_test: Dados de teste (features)
        k: Número de vizinhos
        metric: Métrica de distância ('euclidean', 'manhattan', 'chebyshev', 'mahalanobis')

    Retorna:
        y_pred: Predições para os dados de teste
    """
    y_pred = []

    if metric == 'mahalanobis':
        # Calcula a matriz de covariância
        cov_matrix = np.cov(X_train, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    else:
        inv_cov_matrix = None

    for x_test in X_test:
        if metric == 'mahalanobis':
            # Calculando a distância de Mahalanobis
            distances = np.array(
                [np.sqrt(np.dot(np.dot((x_train - x_test), inv_cov_matrix), (x_train - x_test).T)) for x_train in
                 X_train])
        else:
            # Para outras métricas, usa-se o cdist com a métrica especificada
            distances = cdist([x_test], X_train, metric=metric)[0]

        # Obtém os índices dos k vizinhos mais próximos
        neighbors_idx = np.argsort(distances)[:k]

        # Obtém os rótulos dos k vizinhos
        neighbor_labels = y_train[neighbors_idx]

        # Vota pela maioria
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        y_pred.append(most_common)

    return np.array(y_pred)


def evaluate_knn(X_train, y_train, X_test, y_test, k, metric):
    """
    Avaliação do KNN com base em uma métrica de distância.

    Parâmetros:
        X_train: Dados de treino (features)
        y_train: Rótulos de treino
        X_test: Dados de teste (features)
        y_test: Rótulos reais para os dados de teste
        k: Número de vizinhos
        metric: Métrica de distância ('euclidean', 'manhattan', 'chebyshev', 'mahalanobis')

    Retorna:
        Acurácia do modelo nos dados de teste
    """
    y_pred = knn(X_train, y_train, X_test, k, metric)
    accuracy = np.mean(y_pred == y_test)
    return accuracy


# Carregando o dataset de drogas
dataset_path = r'C:\Users\glaub\Desktop\IA-SI\IA--SI\AV1\Datasets\drug_dataset_ajustado.csv'
drug_data = pd.read_csv(dataset_path)

# Dividindo as colunas de features e rótulos
X = drug_data.drop(columns=['Drug'])  # Droppando a coluna 'Drug' para obter as features
y = drug_data['Drug']  # A coluna 'Drug' será o rótulo

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização de Média Zero e Variância Unitária (obtivemos boa acurácia nesse método anteriormente)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definindo a métrica de distância
metric = 'euclidean'  # Definindo a métrica de distância

# Testando diferentes valores de k (1 a 15)
k_values = range(1, 16)
accuracies = []

for k in k_values:
    acc = evaluate_knn(X_train_scaled, y_train.to_numpy(), X_test_scaled, y_test.to_numpy(), k, metric)
    accuracies.append(acc)

# Plotando o gráfico de acurácias vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Acurácia do KNN para diferentes valores de k')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Melhor valor de k
best_k = k_values[np.argmax(accuracies)]
best_accuracy = np.max(accuracies)

print(f"O melhor valor de k é {best_k} com acurácia de {best_accuracy:.2f}")
