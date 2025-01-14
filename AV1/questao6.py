import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


file_path = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\Ferrari (20.04.23 - 01.05.24).csv"
df = pd.read_csv(file_path)


target = "Close"
feature = "High"


file_path = r"C:\Users\vitor\Downloads\IA.BLACK\IA--SI\AV1\Ferrari (20.04.23 - 01.05.24).csv"
df = pd.read_csv(file_path)

target = "Close"
feature = "High"

scaler = MinMaxScaler()
X = scaler.fit_transform(df[[feature]])
y = df[target].values


model = LinearRegression().fit(X, y)


model = LinearRegression().fit(X, y)

y_pred = model.predict(X)
RSS = np.sum((y - y_pred) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

print(f"Coeficiente (slope): {model.coef_[0]:.4f}")
print(f"Intercepto: {model.intercept_:.4f}")
print(f"RSS: {RSS:.4f}")
print(f"MSE: {MSE:.4f}")
print(f"RMSE: {RMSE:.4f}")
print(f"R²: {R_squared:.4f}")


plt.scatter(X, y, color="blue", label="Dados")
plt.plot(X, y_pred, color="red", label="Reta de Regressão")
plt.title(f"Regressão Linear: {feature} vs {target}")
plt.xlabel(f"{feature} (normalizado)")
plt.ylabel(target)
plt.legend()
plt.show()