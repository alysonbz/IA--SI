import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def k_fold_cross_validation(data, target, model, k=5):
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    fold_sizes = len(data) // k
    folds = [indices[i * fold_sizes:(i + 1) * fold_sizes] for i in range(k)]

    errors = []

    for i in range(k):
        val_indices = folds[i]
        train_indices = np.setdiff1d(indices, val_indices)

        X_train, X_val = data[train_indices], data[val_indices]
        y_train, y_val = target[train_indices], target[val_indices]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        error = np.sqrt(mean_squared_error(y_val, y_pred))
        errors.append(error)

    return errors

def calculate_metrics(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (rss / np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        'RSS': rss,
        'MSE': mse,
        'RMSE': rmse,
        'R^2': r2
    }

file_path = 'Life Expectancy Data.csv'
life_expectancy_data = pd.read_csv(file_path)

data_cleaned = life_expectancy_data.dropna()
data = data_cleaned[['GDP']].values
target = data_cleaned['Life expectancy '].values

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

for name, model in models.items():
    errors = k_fold_cross_validation(data_train, target_train, model, k=5)
    model.fit(data_train, target_train)
    test_pred = model.predict(data_test)
    test_error = np.sqrt(mean_squared_error(target_test, test_pred))

    metrics = calculate_metrics(target_test, test_pred)

    print(f"{name}")
    print(f"Erros por fold: {errors}")
    print(f"Erro médio (validação): {np.mean(errors)}")
    print(f"Erro no conjunto de teste: {test_error}")
    print(f"Métricas: {metrics}\n")

model = LinearRegression()
model.fit(data_train, target_train)
plt.scatter(data, target, color='blue', label='Dados reais')
plt.plot(data, model.predict(data), color='red', label='Ajuste Linear')
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.title('Regressão Linear: GDP vs Life Expectancy')
plt.legend()
plt.show()

print("Embora as diferenças sejam mínimas, a regressão Lasso apresentou o menor erro médio de validação.")
print("Isso pode ser preferido em cenários onde uma ligeira melhoria na validação é desejável.")
print("No entanto, dado o baixo R², é evidente que:O PIB não é suficiente para explicar a variação na expectativa de vida.")
print("É necessário incluir mais variáveis relevantes no modelo para melhorar o desempenho.")