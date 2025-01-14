import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


file_path = "stroke_prediction_dataset_ajustado.csv"
data = pd.read_csv(file_path)

#  Separa variáveis independentes e alvo
X = data.drop(columns=['stroke']).values
y = data['stroke'].values

# Divide o dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Função do KNN
def knn(X_train, y_train, X_test, k, metric):

    if metric == 'mahalanobis':
        cov_matrix = np.cov(X_train, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

    def calculate_distance(x1, x2):
        if metric == 'mahalanobis':
            return mahalanobis(x1, x2, inv_cov_matrix)
        elif metric == 'chebyshev':
            return chebyshev(x1, x2)
        elif metric == 'manhattan':
            return cityblock(x1, x2)
        elif metric == 'euclidean':
            return euclidean(x1, x2)
        else:
            raise ValueError("Métrica de distância inválida.")

    y_pred = []
    for x_test in X_test:
        distances = [calculate_distance(x_test, x_train) for x_train in X_train]
        k_nearest = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_nearest]
        y_pred.append(Counter(k_labels).most_common(1)[0][0])

    return np.array(y_pred)

# métricas
metrics = ['mahalanobis', 'chebyshev', 'manhattan', 'euclidean']
k = 7
print("\nResultados:")
for metric in metrics:
    try:
        y_pred = knn(X_train, y_train, X_test, k, metric)
        accuracy = np.mean(y_pred == y_test)
        print(f"Acurácia com métrica {metric}: {accuracy:.2f}")
    except ValueError as e:
        print(f"Erro com métrica {metric}: {e}")