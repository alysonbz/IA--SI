import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import chebyshev

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

# 1) Importar bibliotecas (já feito acima)

# 2) Carregar o dataset atualizado
file_path = 'waterquality_ajustado.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Arquivo {file_path} não encontrado.")

# Separar recursos e rótulos
X = df.drop('is_safe', axis=1).values
y = df['is_safe'].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3) Normalização logarítmica e verificação de acurácia
def log_normalization(data):
    return np.log1p(data)

X_train_log = log_normalization(X_train)
X_test_log = log_normalization(X_test)

# KNN com normalização logarítmica
k = 7
y_pred_log = knn_predict(X_train_log, y_train, X_test_log, k, chebyshev)
accuracy_log = np.mean(y_pred_log == y_test)

# 4) Normalização de média zero e variância unitária e verificação de acurácia
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN com normalização de média zero e variância unitária
y_pred_scaled = knn_predict(X_train_scaled, y_train, X_test_scaled, k, chebyshev)
accuracy_scaled = np.mean(y_pred_scaled == y_test)

# 5) Printar as acurácias lado a lado
print(f"Acurácia com normalização logarítmica: {accuracy_log:.4f}")
print(f"Acurácia com normalização de média zero e variância unitária: {accuracy_scaled:.4f}")