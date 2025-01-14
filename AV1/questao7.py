import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


from src.utils import load_laptopPrice_dataset
laptop_price = load_laptopPrice_dataset()

laptop_price['rating'] = laptop_price['rating'].str.replace(r'[^0-9.]', '', regex=True)
laptop_price['rating'] = pd.to_numeric(laptop_price['rating'], errors='coerce')
laptop_price = laptop_price.dropna(subset=['rating'])
laptop_price['Number of Ratings'] = pd.to_numeric(laptop_price['Number of Ratings'], errors='coerce')
laptop_price = laptop_price.dropna(subset=['Number of Ratings'])

x = laptop_price[['Number of Ratings']].values
y = laptop_price['Price'].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def calculate_metrics(model, x, y, kf):
    rss_list, mse_list, rmse_list, r2_list = [], [], [], []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        rss = np.sum((y_test - y_pred) ** 2)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)

        rss_list.append(rss)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r_squared)

    return {
        'RSS': np.mean(rss_list),
        'MSE': np.mean(mse_list),
        'RMSE': np.mean(rmse_list),
        'R^2': np.mean(r2_list)
    }

# Modelos a serem testados
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

results = {}
for name, model in models.items():
    results[name] = calculate_metrics(model, x_scaled, y, kf)

for name, metrics in results.items():
    print(f"\n{name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

linear_model = LinearRegression()
linear_model.fit(x_scaled, y)
y_pred = linear_model.predict(x_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=x.flatten(), y=y, label='Dados Reais', alpha=0.6)
plt.plot(x.flatten(), y_pred, color='red', label='Reta de Regressão')
plt.title(f'Regressão Linear\nRSS: {results["Linear Regression"]["RSS"]:.2f}, MSE: {results["Linear Regression"]["MSE"]:.2f}, RMSE: {results["Linear Regression"]["RMSE"]:.2f}, R^2: {results["Linear Regression"]["R^2"]:.2f}')
plt.xlabel('Number of Ratings (Normalizado)')
plt.ylabel('Price')
plt.legend()
plt.show()
