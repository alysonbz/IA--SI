import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Carregar o dataset
df = pd.read_csv('/home/kali/Downloads/gender_classification_ajustado.csv')
print("Dataset carregado.")
columns_needed = ['long_hair', 'forehead_width_cm', 'gender']
df = df[columns_needed]
# . Dividir o dataset em treino e teste
X = df[['long_hair', 'forehead_width_cm']].values
y = df['gender'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# . Implementar o KNN manualmente
def knn_predict(X_train, y_train, X_test, k, metric):
    y_pred = []

    # Calcular matriz de covariância para distância de Mahalanobis
    cov_matrix = np.cov(X_train, rowvar=False) if metric == 'mahalanobis' else None
    inv_cov_matrix = np.linalg.inv(cov_matrix) if cov_matrix is not None else None

    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            if metric == 'mahalanobis':
                dist = mahalanobis(test_point, train_point, inv_cov_matrix)
            elif metric == 'chebyshev':
                dist = chebyshev(test_point, train_point)
            elif metric == 'manhattan':
                dist = cityblock(test_point, train_point)
            elif metric == 'euclidean':
                dist = euclidean(test_point, train_point)
            else:
                raise ValueError("Métrica de distância inválida.")

            distances.append((dist, y_train[i]))

        # Ordenar as distâncias e pegar os k vizinhos mais próximos
        distances.sort(key=lambda x: x[0])
        k_neighbors = [neighbor[1] for neighbor in distances[:k]]

        # Determinar a classe majoritária entre os k vizinhos
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        y_pred.append(most_common)

    return np.array(y_pred)

def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

k_values = range(1, 21)  # Testar k de 1 a 20
metrics = ['mahalanobis', 'chebyshev', 'manhattan', 'euclidean']

best_k_accuracy = {}
best_k_metric = {}

for metric in metrics:
    accuracies = []
    for k in k_values:
        y_pred = knn_predict(X_train_scaled, y_train, X_test_scaled, k, metric)
        accuracy = calculate_accuracy(y_test, y_pred)
        accuracies.append(accuracy)

    best_k = k_values[np.argmax(accuracies)]
    best_accuracy = np.max(accuracies)

    best_k_accuracy[metric] = best_accuracy
    best_k_metric[metric] = best_k

# 7. Exibir resultados
print("\nMelhor k para cada métrica:")
for metric, acc in best_k_accuracy.items():
    print(f"{metric.capitalize()}: {best_k_metric[metric]} com acurácia de {acc:.4f}")

# 8. Plotar o gráfico do melhor k para cada métrica
plt.figure(figsize=(12, 8))
for metric in metrics:
    accuracies = []
    for k in k_values:
        y_pred = knn_predict(X_train_scaled, y_train, X_test_scaled, k, metric)
        accuracy = calculate_accuracy(y_test, y_pred)
        accuracies.append(accuracy)

    plt.plot(k_values, accuracies, label=f'Métrica: {metric}')

plt.title('Acurácia do KNN para diferentes valores de k')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()
