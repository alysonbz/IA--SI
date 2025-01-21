import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar o dataset
classificacao = pd.read_csv('./dataset/dataset_classificacao_ajustado.csv')

# Dividir em X (entradas) e y (saídas)
X = classificacao.drop(columns=['booking_status'], axis=1).values
y = classificacao['booking_status'].values

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Normalização logarítmica
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# Treinando e avaliando para diferentes valores de k
accuracies = []
k_values = range(1, 21)  # Testando k de 1 a 20

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='mahalanobis', metric_params={'VI': np.linalg.inv(np.cov(X_train_log, rowvar=False))})
    knn.fit(X_train_log, y_train)
    y_pred = knn.predict(X_test_log)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotando o gráfico com o melhor k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Acurácia do KNN para diferentes valores de k (distância Mahalanobis)')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()

# Indicar o melhor k
best_k = k_values[np.argmax(accuracies)]
print(f'O melhor valor de k é: {best_k} com uma acurácia de {max(accuracies):.4f}')