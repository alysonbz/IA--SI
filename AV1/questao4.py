from src.utills import diabetes_ajustado_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import chebyshev
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

diabetes = diabetes_ajustado_dataset()

X = diabetes.drop(columns=['Class']).values
y = diabetes['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []

        for i, train_point in enumerate(X_train):
            distance = chebyshev(test_point, train_point)
            distances.append((distance, y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]

        prediction = max(set(k_nearest), key=k_nearest.count)
        predictions.append(prediction)

    return np.array(predictions)

k_values = range(1, 21)
accuracies = []

for k in k_values:
    y_pred = knn_predict(X_train_scaled, y_train, X_test_scaled, k)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Acurácia')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Melhor k = {best_k}')
plt.title('Acurácia do KNN para diferentes valores de k')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.legend()
plt.grid()
plt.show()

print(f"\nMelhor valor de k: {best_k}")
print(f"Acurácia com k={best_k}: {best_accuracy:.2f}")