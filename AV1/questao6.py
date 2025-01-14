import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# carregar o dataset novamente
file_path = r'C:\Users\Administrator\Downloads\DataSet\Acer (2000-2024).csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])

# Selecionar apenas colunas numéricas
df_numerico = df.select_dtypes(include=['float64', 'int64'])

# Selecionar o atributo mais relevante "high"
X = df_numerico[['High']]  
y = df_numerico['Close']

# Criar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

#metricas 
RSS = np.sum((y - y_pred) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

print(f"RSS (Residual Sum of Squares): {RSS}")
print(f"MSE (Mean Squared Error): {MSE}")
print(f"RMSE (Root Mean Squared Error): {RMSE}")
print(f"R² (Coeficiente de Determinação): {R_squared}")

#gráficos
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Reta de regressão')
plt.xlabel('High')
plt.ylabel('Close')
plt.title('Reta de Regressão Linear - Predição de Preço de Fechamento (Close)')
plt.legend()
plt.show()