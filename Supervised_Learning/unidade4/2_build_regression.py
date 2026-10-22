from src.utils import load_sales_clean_dataset
import pandas as pd

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from sklearn.linear_model import LinearRegression


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)


print("**5 primriras predições: \n", predictions[:5])
#print(sales_df.head())#inicio od dataset

#FUNCIONANDO..