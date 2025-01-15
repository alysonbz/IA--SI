import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Importando o dataset
regressao = pd.read_csv('./dataset/dataset_regressao_ajustado.csv')

# 2. Escolhendo o atributo mais relevante baseado na análise de correlação
X = regressao[['smoker']]  # Usando 'smoker' como variável independente
y = regressao['charges']  # Variável dependente

# 3. Normalização (aplicando a melhor normalização, StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 6. Previsões
y_pred = modelo.predict(X_test)

# 7. Calcular as métricas
# RSS (Residual Sum of Squares)
RSS = np.sum((y_test - y_pred) ** 2)

# MSE (Mean Squared Error)
MSE = mean_squared_error(y_test, y_pred)

# RMSE (Root Mean Squared Error)
RMSE = np.sqrt(MSE)

# R² (Coeficiente de Determinação)
R2 = r2_score(y_test, y_pred)

# Exibir as métricas
print(f"\nRSS: {RSS}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R²: {R2}")

# 8. Plotar o gráfico com a reta de regressão
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Nuvem de pontos')
plt.plot(X_test, y_pred, color='red', label='Reta de regressão')
plt.title('Reta de Regressão Linear')
plt.xlabel('Smoker (Fumantes)')
plt.ylabel('Charges')
plt.legend()
plt.show()