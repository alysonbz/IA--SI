import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar o dataset
data = pd.read_csv("stroke_prediction_dataset_ajustado.csv")
X = data.drop(columns=['stroke']).values
y = data['stroke'].values

# Função KNN otimizada
def knn(X_train, y_train, X_test, k):
    distances = cdist(X_test, X_train, 'euclidean')
    k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
    k_nearest_labels = y_train[k_nearest_indices]
    y_pred = [np.bincount(labels).argmax() for labels in k_nearest_labels]
    return np.array(y_pred)

# Dividir os dados e normalizar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testar diferentes valores de k e calcular a acurácia
k_values = range(1, 21)
accuracies = []

for k in k_values:
    y_pred = knn(X_train_scaled, y_train, X_test_scaled, k)
    accuracies.append(np.mean(y_pred == y_test))

# mostrar um gráfico
plt.plot(k_values, accuracies, marker='o')
plt.title("Acurácia do KNN para diferentes valores de k")
plt.xlabel("Valor de k")
plt.ylabel("Acurácia")
plt.grid(True)
plt.show() 

# Exibir o melhor k
best_k = k_values[np.argmax(accuracies)]
print(f"O melhor valor de k é: {best_k} com acurácia de {max(accuracies):.2f}")