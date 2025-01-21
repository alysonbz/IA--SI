import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

file_path = 'Life Expectancy Data.csv'
life_expectancy_data = pd.read_csv(file_path)

life_expectancy_data_cleaned = life_expectancy_data.dropna()

target = 'Life expectancy '
most_significant_feature = 'Schooling'

X = life_expectancy_data_cleaned[[most_significant_feature]]
y = life_expectancy_data_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rss = np.sum((y_test - y_pred) ** 2)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Residual Sum of Squares (RSS): {rss}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R^2): {r2}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test[most_significant_feature], y=y_test, alpha=0.7, label='Actual')
plt.plot(X_test[most_significant_feature], y_pred, color='red', label='Regression Line')
plt.xlabel(most_significant_feature)
plt.ylabel('Life Expectancy')
plt.title('Regression Line with Actual Data')
plt.legend()
plt.show()
