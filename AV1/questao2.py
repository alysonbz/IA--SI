from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean


def calculate_distance(metric, point1, point2, cov_matrix=None):
    if metric == "mahalanobis":
        return mahalanobis(point1, point2, cov_matrix)
    elif metric == "chebyshev":
        return chebyshev(point1, point2)
    elif metric == "manhattan":
        return cityblock(point1, point2)
    elif metric == "euclidean":
        return euclidean(point1, point2)
    else:
        raise ValueError("Métrica desconhecida.")

def knn(X_train, y_train, X_test, k, metric):
    predictions = []
    cov_matrix_inv = None
    X_train_array = np.array(X_train, dtype=np.float64)  # Garante conversão numérica

    if metric == "mahalanobis":
        cov_matrix = np.cov(X_train_array.T)
        cov_matrix_inv = np.linalg.inv(cov_matrix)

    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train_array):
            dist = calculate_distance(metric, test_point, train_point, cov_matrix_inv)
            distances.append((dist, y_train[i]))

        distances = sorted(distances, key=lambda x: x[0])[:k]
        neighbors_classes = [d[1] for d in distances]
        predicted_class = max(set(neighbors_classes), key=neighbors_classes.count)
        predictions.append(predicted_class)

    return np.array(predictions)

def evaluate_knn(X_train, y_train, X_test, y_test, k, metric):
    X_train_array = X_train.to_numpy(dtype=np.float64)
    X_test_array = X_test.to_numpy(dtype=np.float64)
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    y_pred = knn(X_train_array, y_train_array, X_test_array, k, metric)
    accuracy = np.mean(y_pred == y_test_array)
    return accuracy


dataset_path = r'C:\Users\glaub\Desktop\IA-SI\IA--SI\AV1\Datasets\drug_dataset_ajustado.csv'
drug_data = pd.read_csv(dataset_path)

X = drug_data.drop(columns=['Drug'])
y = drug_data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

metrics = ["mahalanobis", "chebyshev", "manhattan", "euclidean"]
k = 7
results = {}

for metric in metrics:
    acc = evaluate_knn(X_train, y_train, X_test, y_test, k, metric)
    results[metric] = acc
    print(f"Acurácia usando {metric}: {acc:.2f}")
