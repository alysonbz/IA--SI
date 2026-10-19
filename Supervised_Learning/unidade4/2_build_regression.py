from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from sklearn.linear_model import LinearRegression


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_bmi, y)

# Make predictions
predictions = preg.redict(X_bmi)


print(predictions.head())