from src.utills import houses_sales_processed_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

houses = houses_sales_processed_dataset()

most_relevant_feature = 'sqft_living'
target = 'price'

X = houses[[most_relevant_feature]].values
y = houses[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

RSS = np.sum((y_test - y_pred) ** 2)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y_test, y_pred)

print(f"RSS: {RSS:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(f"R²: {R_squared:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados reais', alpha=0.5)
plt.plot(X_test, model.predict(X_test_scaled), color='red', label='Reta de regressão')
plt.title('Reta de Regressão Linear')
plt.xlabel(most_relevant_feature)
plt.ylabel('Preço')
plt.legend()
plt.grid()
plt.show()

