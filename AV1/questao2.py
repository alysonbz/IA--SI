from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
import numpy as np
import pandas as pd

def load_combined_data():
    return pd.read_csv('credit_risk_data.csv')
credit_risk_data = load_combined_data()

#Features e Target
X = credit_risk_data.drop(['label'], axis=1).values  # Features
y = credit_risk_data['label'].values  #Target

#Divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#KNN Manual
def knn(X_train, y_train, X_test, k, dist_metric):
    predictions = []
    for test_point in X_test:
        #Calcular distâncias
        if dist_metric == 'euclidean':
            dists = np.array([euclidean(x, test_point) for x in X_train])
        elif dist_metric == 'manhattan':
            dists = np.array([cityblock(x, test_point) for x in X_train])
        elif dist_metric == 'chebyshev':
            dists = np.array([chebyshev(x, test_point) for x in X_train])
        elif dist_metric == 'mahalanobis':
            vi = np.cov(X_train.T)  #Matriz de covariância
            inv_vi = np.linalg.inv(vi)  #Inversa da matriz de covariância
            dists = np.array([mahalanobis(test_point, x, inv_vi) for x in X_train])

        #K vizinhos mais próximos
        k_indices = np.argsort(dists)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]

        #Predição pela classe mais comum
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)

    return np.array(predictions)

#Cálculo da acurácia
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

k = 7
metrics = ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']
accuracies = {}

for metric in metrics:
    y_pred = knn(X_train, y_train, X_test, k, metric)
    acc = accuracy(y_test, y_pred)
    accuracies[metric] = acc

#Printar as acurácias
print("\nAcurácias para diferentes métricas de distância:")
for metric, acc in accuracies.items():
    print(f"{metric.capitalize()}: {acc:.9f}")