# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = r'C:\Users\adrie\OneDrive\Documentos\IA-SI_2\IA--SI\AV1\possum_ajustado.csv'
data = pd.read_csv(file_path)

# Separar os dados em variáveis independentes (X) e dependentes (y)
X = data[["hdlngth"]]
y = data["age"]

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

#Modelo de regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular RSS, MSE, RMSE e R²
rss = np.sum((y_test - y_pred) ** 2)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

print(f"RSS: {rss:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r_squared:.2f}")

# Plotar a reta de regressão com a nuvem de pontos
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="Dados reais")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Reta de regressão")
plt.title(f"Regressão Linear: hdlngth vs Age")
plt.xlabel('hdlngth')
plt.ylabel('Age')
plt.legend()
plt.grid()
plt.show()