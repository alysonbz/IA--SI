from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression

y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

from sklearn.linear.model import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X,y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])
print(y[0:5])