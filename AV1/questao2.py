import numpy as np
from src.utills import diabetes_ajustado_dataset
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean

diabetes = diabetes_ajustado_dataset()

X = diabetes.drop(columns=['Class']).values
y = diabetes['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def knn_predict(X_train, y_train, X_test, k, metric):
    predictions = []
    for test_point in X_test:
        distances = []

        for i, train_point in enumerate(X_train):
            if metric == "mahalanobis":
                VI = np.linalg.inv(np.cov(X_train.T))
                distance = mahalanobis(test_point, train_point, VI)
            elif metric == "chebyshev":
                distance = chebyshev(test_point, train_point)
            elif metric == "manhattan":
                distance = cityblock(test_point, train_point)
            elif metric == "euclidean":
                distance = euclidean(test_point, train_point)
            distances.append((distance, y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]

        prediction = max(set(k_nearest), key=k_nearest.count)
        predictions.append(prediction)

    return np.array(predictions)

k = 7
metrics = ["mahalanobis", "chebyshev", "manhattan", "euclidean"]
accuracies = {}

for metric in metrics:
    y_pred = knn_predict(X_train, y_train, X_test, k, metric)
    accuracy = np.mean(y_pred == y_test)
    accuracies[metric] = accuracy
    print(f"Acurácia usando a métrica {metric}: {accuracy:.2f}")

print("\nComparação de acurácias:")
for metric, acc in accuracies.items():
    print(f"{metric.capitalize()}: {acc:.2f}")