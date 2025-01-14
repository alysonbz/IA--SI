import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Passo 2: Carregar o dataset
dataset_path = 'bodyfat.csv'  # Ajuste o caminho conforme necessário
df = pd.read_csv(dataset_path)

# Passo 3: Verificar dados ausentes
print(df.isnull().sum())  # Verifique se há NaNs

# Remover valores ausentes (se houver)
df = df.dropna()

# Passo 4: Separar o alvo (BodyFat) e as variáveis independentes
X = df.drop("BodyFat", axis=1)  # Remover a coluna 'BodyFat'
y = df["BodyFat"]  # 'BodyFat' será nosso alvo

# Passo 5: Normalizar os dados (usar StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Passo 6: Implementar a regressão linear
# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Fazer previsões
y_pred = regressor.predict(X_test)

# Passo 7: Calcular as métricas de desempenho
# Calcular o RSS, MSE, RMSE e R²
rss = np.sum((y_test - y_pred) ** 2)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

# Exibir os resultados
print(f"RSS: {rss:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r_squared:.4f}")

# Passo 8: Plotar o gráfico (retas de regressão versus pontos reais)
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Regressão Linear: Predições vs. Valores Reais')
plt.xlabel('Valores Reais de BodyFat')
plt.ylabel('Predições de BodyFat')
plt.show()