import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('car_price.csv')
# 1) Separar o atributo mais relevante e o alvo
X = df[['Prod. year']].values  # Atributo mais relevante
y = df['Price'].values         # Alvo (Price)

# 2) Dividir os dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3) Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# 4) Fazer predições
y_pred = model.predict(X_test)

# 5) Calcular métricas
rss = np.sum((y_test - y_pred) ** 2)  # Soma dos resíduos ao quadrado
mse = mean_squared_error(y_test, y_pred)  # Erro médio quadrado
rmse = np.sqrt(mse)  # Raiz do erro médio quadrado
r_squared = r2_score(y_test, y_pred)  # R²

# 6) Exibir as métricas
print("Métricas de desempenho:")
print(f"RSS: {rss:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r_squared:.4f}")

# 7) Visualizar a reta de regressão
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados reais')
plt.plot(X_test, y_pred, color='red', label='Reta de regressão')
plt.title('Regressão Linear - Prod. year vs Price')
plt.xlabel('Prod. year')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()