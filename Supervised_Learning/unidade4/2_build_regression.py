from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from sklearn.linear_model import LinearRegression


y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()

# Fit the model to the data
<<<<<<< HEAD
reg.fit(X, y)
=======
reg.fit(X,y)
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

# Make predictions
predictions = reg.predict(X)

<<<<<<< HEAD
print(predictions[:5])
print(y[0:5])
=======
print(y[:5])
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1
