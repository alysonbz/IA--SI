import pandas as pd
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import mean_squared_error
from sklearn.metrics import mean_squared_error, root_mean_squared_error

sales_df = load_sales_clean_dataset()

# Remove linhas com valores ausentes em sales e influencer
sales_df = sales_df.dropna(subset=["sales", "influencer"])

# Create X and y arrays
# "influencer" é categórica, então convertemos para colunas numéricas (dummies)
X = pd.get_dummies(sales_df.drop(columns=["sales"]), drop_first=True)
y = sales_df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg_all = LinearRegression()

# Fit the model to the data
reg_all.fit(X_train, y_train)

# Make predictions
y_pred = reg_all.predict(X_test)
print(f"Predictions: {y_pred[:2]}, Actual Values: {y_test[:2].values}")

# Compute R-squared
r_squared = reg_all.score(X_test, y_test)

# Compute RMSE
rmse = root_mean_squared_error(y_test, y_pred)

# Print the metrics
print(f"R^2: {r_squared}")
print(f"RMSE: {rmse}")