import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis, cityblock, chebyshev, euclidean
from sklearn.model_selection import train_test_split
from collections import Counter

dataset = pd.read_csv('Dataset_coletado.csv')

X = dataset.drop(columns=['blue'])
y = dataset['blue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def knn(X_train, y_train, X_test, k, distance_func):
    predictions = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            distance = distance_func(test_point, train_point)
            distances.append((distance, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        k_nearest_labels = [label for _, label in k_nearest]
        prediction = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(prediction)
    return predictions

X_train_array = X_train.to_numpy()
X_test_array = X_test.to_numpy()
y_train_array = y_train.to_numpy()

def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

distance_functions = {
    'Mahalanobis': lambda x, y: mahalanobis(x, y, np.linalg.inv(np.cov(X_train_array, rowvar=False))),
    'Chebyshev': chebyshev,
    'Manhattan': cityblock,
    'Euclidean': euclidean
}
results = {}

for name, func in distance_functions.items():
    print(f"Calculando KNN usando {name}...")
    y_pred = knn(X_train_array, y_train_array, X_test_array, k=7, distance_func=func)
    accuracy = compute_accuracy(y_test.to_numpy(), y_pred)
    results[name] = accuracy
    print(f"Acurácia usando {name}: {accuracy:.2f}")

print("\nResultados finais:")
for distance, acc in results.items():
    print(f"{distance}: {acc:.2f}")