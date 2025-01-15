import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o dataset
df = pd.read_csv('/home/kali/Downloads/kc_house_data.csv')

X = df[['sqft_living']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

RSS = np.sum((y_test - y_pred) ** 2)  # Residual Sum of Squares
MSE = mean_squared_error(y_test, y_pred)  # Mean Squared Error
RMSE = np.sqrt(MSE)  # Root Mean Squared Error
R_squared = r2_score(y_test, y_pred)  # Coeficiente de determinação

print(f"RSS: {RSS}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R^2: {R_squared}")

# Ordenar os dados para um gráfico da reta de regressão
X_test_sorted = X_test.sort_values(by='sqft_living')  # Ordenar X_test por 'sqft_living'
y_pred_sorted = model.predict(X_test_sorted)  # Previsão com base nos valores ordenados

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Dados reais', color='blue')  # Scatter dos dados reais
plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Reta de regressão', linewidth=2)  # Reta de regressão
plt.title('Reta de Regressão Linear: sqft_living vs price')
plt.xlabel('Área habitável (sqft)')
plt.ylabel('Preço de venda')
plt.legend()
plt.show()