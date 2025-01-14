import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o dataset ajustado
data = pd.read_csv("stroke_prediction_dataset_ajustado.csv")

# Separar variáveis independentes e alvo
X = data.drop(columns=['stroke']).values  
y = data['stroke'].values

# Função KNN
def knn(X_train, y_train, X_test, k=7):
    y_pred = []
    for x_test in X_test:
        distances = [euclidean(x_test, x_train) for x_train in X_train]
        k_nearest = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_nearest]
        y_pred.append(max(set(k_labels), key=k_labels.count))  # Votação majoritária
    return np.array(y_pred)

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização logarítmica
X_train_log = np.log1p(X_train.astype(float))  
X_test_log = np.log1p(X_test.astype(float))  

# Normalização de média zero e variância unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Avaliar a acurácia com normalização logarítmica
y_pred_log = knn(X_train_log, y_train, X_test_log)
accuracy_log = np.mean(y_pred_log == y_test)

# Avaliar a acurácia com média zero e variância unitária
y_pred_scaled = knn(X_train_scaled, y_train, X_test_scaled)
accuracy_scaled = np.mean(y_pred_scaled == y_test)

print(f"Acurácia com média zero e variância unitária: {accuracy_scaled:.2f}")