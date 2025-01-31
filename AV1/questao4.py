import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import chebyshev
import matplotlib.pyplot as plt

# Função KNN
def knn_predict(X_train, y_train, X_test, k, distance_metric):
    y_pred = []
    for test_point in X_test:
        distances = []
        for train_point, train_label in zip(X_train, y_train):
            dist = distance_metric(test_point, train_point)
            distances.append((dist, train_label))
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]
        y_pred.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
    return np.array(y_pred)

# Função para calcular a distância de Chebyshev
def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

# 1) Importar bibliotecas (já feito acima)

# 2) Carregar o dataset atualizado
file_path = 'waterquality_ajustado.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Arquivo {file_path} não encontrado.")
    exit()

# Separar recursos e rótulos
X = df.drop('is_safe', axis=1).values
y = df['is_safe'].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3) Normalização com média zero e variância unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Encontrar o melhor valor de k
k_values = range(6, 10)
accuracies = []

for k in k_values:
    y_pred = knn_predict(X_train_scaled, y_train, X_test_scaled, k, chebyshev_distance)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

# Melhor valor de k
best_k = k_values[np.argmax(accuracies)]

# 5) Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Acurácia do KNN em função de k (distância de Chebyshev)')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.grid()
plt.axvline(best_k, color='r', linestyle='--', label=f'Melhor k: {best_k}')
plt.legend()
plt.show()

print(f"Melhor valor de k: {best_k} com acurácia de {max(accuracies):.4f}")
