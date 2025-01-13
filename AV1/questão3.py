import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis, chebyshev, cityblock, euclidean
from sklearn.preprocessing import StandardScaler

# 1. Importar as bibliotecas necessárias
# (importações feitas acima)

# 2. Carregar o dataset atualizado ou original
try:
    # Tentar carregar o dataset atualizado
    df = pd.read_csv("gender_classification_adjusted.csv")
    print("Dataset atualizado carregado.")
except FileNotFoundError:
    # Carregar o dataset original caso o atualizado não esteja disponível
    url = "https://www.kaggleusercontent.com/datasets/elakiricoder/gender-classification-dataset/data.csv"
    df = pd.read_csv(url)
    print("Dataset original carregado.")

# Manter apenas as colunas relevantes
columns_needed = ['height', 'weight', 'gender']
df = df[columns_needed]

# Converter classes para valores numéricos, se necessário
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Dividir o dataset em treino e teste
X = df[['height', 'weight']].values
y = df['gender'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Implementar o KNN manual com a melhor métrica
def knn_predict(X_train, y_train, X_test, k, metric):
    """
    Implementa o algoritmo KNN de forma manual.
    """
    y_pred = []

    # Calcular matriz de covariância para distância de Mahalanobis
    cov_matrix = np.cov(X_train, rowvar=False) if metric == 'mahalanobis' else None
    inv_cov_matrix = np.linalg.inv(cov_matrix) if cov_matrix is not None else None

    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            if metric == 'mahalanobis':
                dist = mahalanobis(test_point, train_point, inv_cov_matrix)
            elif metric == 'chebyshev':
                dist = chebyshev(test_point, train_point)
            elif metric == 'manhattan':
                dist = cityblock(test_point, train_point)
            elif metric == 'euclidean':
                dist = euclidean(test_point, train_point)
            else:
                raise ValueError("Métrica de distância inválida.")

            distances.append((dist, y_train[i]))

        # Ordenar as distâncias e pegar os k vizinhos mais próximos
        distances.sort(key=lambda x: x[0])
        k_neighbors = [neighbor[1] for neighbor in distances[:k]]

        # Determinar a classe majoritária entre os k vizinhos
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        y_pred.append(most_common)

    return np.array(y_pred)


# Melhor métrica de distância identificada
best_metric = "euclidean"  # Substituir pela métrica identificada no exercício anterior
k = 7

# 3. Normalização logarítmica
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# Predição e acurácia com normalização logarítmica
y_pred_log = knn_predict(X_train_log, y_train, X_test_log, k, best_metric)
accuracy_log = accuracy_score(y_test, y_pred_log)

# 4. Normalização de média zero e variância unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Predição e acurácia com normalização padrão
y_pred_scaled = knn_predict(X_train_scaled, y_train, X_test_scaled, k, best_metric)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

# 5. Comparar acurácias
print("\nComparação das acurácias:")
print(f"Acurácia com normalização logarítmica: {accuracy_log:.4f}")
print(f"Acurácia com normalização de média zero e variância unitária: {accuracy_scaled:.4f}")