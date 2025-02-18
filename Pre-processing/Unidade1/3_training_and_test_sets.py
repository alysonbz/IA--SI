import numpy as np
from src.utils import diabetes_ajustado_dataset
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import chebyshev
from sklearn.preprocessing import StandardScaler

diabetes = diabetes_ajustado_dataset()

X = diabetes.drop(columns=['Class']).values
y = diabetes['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

k = 7
y_pred_original = knn_predict(X_train, y_train, X_test, k)
accuracy_original = np.mean(y_pred_original == y_test)

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)
y_pred_log = knn_predict(X_train_log, y_train, X_test_log, k)
accuracy_log = np.mean(y_pred_log == y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_pred_scaled = knn_predict(X_train_scaled, y_train, X_test_scaled, k)
accuracy_scaled = np.mean(y_pred_scaled == y_test)

print("Acurácias:")
print(f"Sem normalização: {accuracy_original:.2f}")
print(f"Com normalização logarítmica: {accuracy_log:.2f}")
print(f"Com normalização de média zero e variância unitária: {accuracy_scaled:.2f}")