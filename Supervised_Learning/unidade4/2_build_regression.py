from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from sklearn.import


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = ____()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(0.5)