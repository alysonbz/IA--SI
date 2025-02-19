import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Carregar o dataset
classificacao = pd.read_csv('./dataset/dataset_classificacao_ajustado.csv')

#Dividir em X (entradas) e y (saídas)
X = classificacao.drop(columns=['booking_status'], axis=1).values
y = classificacao['booking_status'].values

#Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

#Normalização logarítmica
log_transformer = FunctionTransformer(np.log1p, validate=True)
X_train_log = log_transformer.fit_transform(X_train)
X_test_log = log_transformer.transform(X_test)

#Normalização z-score
scaler = StandardScaler()
X_train_zscore = scaler.fit_transform(X_train)
X_test_zscore = scaler.transform(X_test)

#Limitar o número de amostras para 5000
X_train_limited = X_train[:5000]
y_train_limited = y_train[:5000]
X_test_limited = X_test[:5000]
y_test_limited = y_test[:5000]

#Instanciando o KNeighborsClassifier com Mahalanobis como a métrica
knn = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis', metric_params={'VI': np.linalg.inv(np.cov(X_train_log, rowvar=False))})

#Avaliar com normalização logarítmica
knn.fit(X_train_log, y_train)
y_pred_log = knn.predict(X_test_log)
accuracy_log = accuracy_score(y_test, y_pred_log)

#Avaliar com normalização z-score
knn.fit(X_train_zscore, y_train)
y_pred_zscore = knn.predict(X_test_zscore)
accuracy_zscore = accuracy_score(y_test, y_pred_zscore)

#Comparar resultados
print("Acurácias:")
print(f"Normalização Logarítmica: {accuracy_log:.4f}")
print(f"Normalização Z-score: {accuracy_zscore:.4f}")