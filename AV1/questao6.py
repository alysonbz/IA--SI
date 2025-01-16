import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def load_activision_blizzard():
    return pd.read_csv('activision_blizzard.csv')
activision = load_activision_blizzard()

#Métricas
def compute_RSS(predictions, y):
    RSS = np.sum(np.square(predictions - y))
    return RSS

def compute_MSE(predictions, y):
    MSE = np.mean(np.square(predictions - y))
    return MSE

def compute_RMSE(predictions, y):
    RMSE = np.sqrt(compute_MSE(predictions, y))
    return RMSE

def compute_R_squared(predictions, y):
    total = np.sum(np.square(y - np.mean(y)))
    residual = compute_RSS(predictions, y)
    r_squared = 1 - (residual / total)
    return r_squared

#Colunas relevantes
activision_relevant = activision[['High', 'Close']]

#Normalizar os dados
scaler = StandardScaler()
X = activision_relevant[['High']]
y = activision_relevant['Close']

#Normalizar os dados
X_scaled = scaler.fit_transform(X)

#Divisão os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#Regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Calculo das métricas
rss = compute_RSS(y_pred, y_test)  # Residual Sum of Squares
mse = compute_MSE(y_pred, y_test)  # Mean Squared Error
rmse = compute_RMSE(y_pred, y_test)  # Root Mean Squared Error
r_squared = compute_R_squared(y_pred, y_test)  # R-squared

#Printando das métricas
print(f'RSS: {rss:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R-squared: {r_squared:.2f}')

#Plotando o gráfico da reta de regressão
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Reta de Regressão')
plt.title('Regressão Linear: Previsão de Preço de Fechamento (Close) com Atributo High')
plt.xlabel('High')
plt.ylabel('Close')
plt.legend()
plt.show()

#Melhor K
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 12)

for neighbor in neighbors:
    # Regressor KNN
    knn = KNeighborsRegressor(n_neighbors=neighbor)
    knn.fit(X_train, y_train)

    # Cálculo da acurácia
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

# Acurácias
print("\nAcurácia no Treino: ", train_accuracies)
print("Acurácia no Teste: ", test_accuracies)

# Melhor K
best_k = max(test_accuracies, key=test_accuracies.get)
best_accuracy = test_accuracies[best_k]
print(f'\nO melhor valor de K é: {best_k} com uma acurácia de: {best_accuracy:.6f}')

# Plotando as acurácias
plt.figure(figsize=(10, 6))
plt.title("KNN: Variação do Número de Vizinhos")
plt.plot(neighbors, train_accuracies.values(), label="Acurácia de Treinamento", marker='o')
plt.plot(neighbors, test_accuracies.values(), label="Acurácia de Teste", marker='o')
plt.legend()
plt.xlabel("Número de Vizinhos (k)")
plt.ylabel("Acurácia")
plt.xticks(neighbors)
plt.grid()
plt.show()