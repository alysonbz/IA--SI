import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, mahalanobis, chebyshev, cityblock
from collections import Counter

# 1. Importar as bibliotecas necessárias
# (importações acima já cobrem esta etapa)

# 2. Carregar o dataset atualizado ou original
try:
    # Tentar carregar o dataset atualizado
    df = pd.read_csv("gender_classification_ajustado.csv")
    print("Dataset atualizado carregado.")
except FileNotFoundError:
    # Carregar o dataset original caso o atualizado não esteja disponível
    df = pd.read_csv('/home/kali/Downloads/gender_classification_v7.xls')
    print("Dataset original carregado.")

# Manter apenas as colunas relevantes
columns_needed = ['long_hair', 'forehead_width_cm', 'gender']
df = df[columns_needed]

# Converter classes para valores numéricos, se necessário
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Dividir o dataset em treino e teste
X = df[['long_hair', 'forehead_width_cm']].values
y = df['gender'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Normalizar o conjunto de dados com a melhor normalização
# Supondo que a melhor normalização seja média zero e variância unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Função para implementar o KNN manual
def knn_predict(X_train, y_train, X_test, k, metric):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            if metric == 'euclidean':
                dist = euclidean(test_point, train_point)
            else:
                raise ValueError("Somente 'euclidean' é implementado neste exemplo.")
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_neighbors = [neighbor[1] for neighbor in distances[:k]]
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)

# 4. Buscar o melhor valor de k e plotar o gráfico
k_values = range(1, 21)  # Testar valores de k de 1 a 20
accuracies = []

for k in k_values:
    y_pred = knn_predict(X_train_scaled, y_train, X_test_scaled, k, metric='euclidean')
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotar o gráfico
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title("Acurácia do KNN para diferentes valores de k")
plt.xlabel("Valor de k")
plt.ylabel("Acurácia")
plt.xticks(k_values)
plt.grid()
plt.show()

# Identificar o melhor k
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\nMelhor valor de k: {best_k}")
print(f"Acurácia com k = {best_k}: {best_accuracy:.4f}")