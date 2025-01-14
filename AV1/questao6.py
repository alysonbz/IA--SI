import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Carregar o dataset atualizado
from src.utils import load_laptopPrice_dataset
laptop_price = load_laptopPrice_dataset()

# Removendo NaNs das colunas relevantes
laptop_price = laptop_price.dropna(subset=['Number of Ratings', 'Number of Reviews'])

# 2. Normalização do conjunto de dados
# Comparando diferentes normalizações
scalers = {'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler()}
best_scaler = None
best_r_squared = -np.inf

for scaler_name, scaler in scalers.items():
    X_normalized = scaler.fit_transform(laptop_price['Number of Ratings'].values.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_normalized, laptop_price['Number of Reviews'])
    r_squared = model.score(X_normalized, laptop_price['Number of Reviews'])
    if r_squared > best_r_squared:
        best_r_squared = r_squared
        best_scaler = scaler

# Aplicando o melhor scaler
scaler = best_scaler
X = scaler.fit_transform(laptop_price['Number of Ratings'].values.reshape(-1, 1))
y = laptop_price['Number of Reviews'].values

# 3. Treinamento do modelo de regressão linear
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 4. Cálculo das métricas
residuals = y - y_pred
RSS = np.sum(residuals**2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

print(f"Melhor Normalização: {type(scaler).__name__}")
print(f'RSS: {RSS}')
print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')
print(f'R²: {R_squared}')

# 5. Gráfico da reta de regressão
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.flatten(), y=y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Reta de regressão')
plt.title('Regressão Linear: Número de Avaliações vs Número de Resenhas')
plt.xlabel('Number of Ratings (Normalizado)')
plt.ylabel('Number of Reviews')
plt.legend()
plt.show()