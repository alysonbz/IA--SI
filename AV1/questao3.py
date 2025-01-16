import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

def load_combined_data():
    return pd.read_csv('credit_risk_data.csv')

credit_risk_data = load_combined_data()

# Features e Target
X = credit_risk_data.drop(['label'], axis=1).values
y = credit_risk_data['label'].values

# Divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Acurácia sem normalização usando distância Mahalanobis
cov_matrix = np.cov(X_train.T)

# Verificar se a matriz de covariância é singular
try:
    inv_cov_matrix = np.linalg.inv(cov_matrix)
except np.linalg.LinAlgError:
    print("A matriz de covariância é singular. Adicionando regularização.")
    # Adicionando um pequeno valor à diagonal para regularização
    inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-10)

# Acurácia sem normalização
knn_no_norm = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis',
                                   metric_params={'VI': inv_cov_matrix})
knn_no_norm.fit(X_train, y_train)
y_pred_no_norm = knn_no_norm.predict(X_test)
accuracy_no_norm = accuracy_score(y_test, y_pred_no_norm)

# Normalização Logarítmica
# Substituir valores zero por um pequeno valor positivo
X_train_log = np.where(X_train <= 0, 1e-10, X_train)
X_test_log = np.where(X_test <= 0, 1e-10, X_test)

# Aplicar log1p
X_train_log = np.log1p(X_train_log)
X_test_log = np.log1p(X_test_log)

# Recalcular a matriz de covariância e sua inversa para os dados logarítmicos
cov_matrix_log = np.cov(X_train_log.T)
try:
    inv_cov_matrix_log = np.linalg.inv(cov_matrix_log)
except np.linalg.LinAlgError:
    print("A matriz de covariância logarítmica é singular. Adicionando regularização.")
    inv_cov_matrix_log = np.linalg.inv(cov_matrix_log + np.eye(cov_matrix_log.shape[0]) * 1e-10)

# Acurácia para normalização logarítmica usando distância Mahalanobis
knn_log = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis',
                               metric_params={'VI': inv_cov_matrix_log})
knn_log.fit(X_train_log, y_train)
y_pred_log = knn_log.predict(X_test_log)
accuracy_log = accuracy_score(y_test, y_pred_log)

# Normalização de Média Zero e Variância Unitária
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Recalcular a matriz de covariância e sua inversa para os dados escalados
cov_matrix_scaled = np.cov(X_train_scaled.T)
try:
    inv_cov_matrix_scaled = np.linalg.inv(cov_matrix_scaled)
except np.linalg.LinAlgError:
    print("A matriz de covariância escalada é singular. Adicionando regularização.")
    inv_cov_matrix_scaled = np.linalg.inv(cov_matrix_scaled + np.eye(cov_matrix_scaled.shape[0]) * 1e-10)

# Acurácia para normalização de média zero e variância unitária usando distância Mahalanobis
knn_scaled = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis',
                                  metric_params={'VI': inv_cov_matrix_scaled})
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

# Print das acurácias
print()
print(f'Acurácia sem Normalização: {accuracy_no_norm:.4f}')
print(f'Acurácia com Normalização Logarítmica: {accuracy_log:.4f}')
print(f'Acurácia com Normalização de Média Zero e Variância Unitária: {accuracy_scaled:.4f}')

# Gráfico comparativo das acurácias
labels = ['Sem Normalização', 'Logarítmica', 'Média Zero e Variância Unitária']
accuracies = [accuracy_no_norm, accuracy_log, accuracy_scaled]

plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácias')
plt.show()