from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sales_df = load_sales_clean_dataset()

<<<<<<< HEAD
# Create X and y arrays
X = sales_df.drop(["sales","influencer"],axis=1)
y = sales_df["sales"].values
=======
# Import mean_squared_error
from sklearn.metrics import mean_square_error

# Create X and y arrays
X = sales_df.dropna(["sales","influencer" ],axis=1)
y = sales_df.dropna["sales"].values
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
<<<<<<< HEAD
reg = LinearRegression()
=======
reg = LinearRegression
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Compute R-squared
<<<<<<< HEAD
r_squared = reg.score(X_test,y_test)

# Compute RMSE
rmse = mean_squared_error(y_pred, y_test, squared=False)
=======
r_squared = reg.square(X_test, y_test)

# Compute RMSE
rmse = mean_square_error(y_pred, y_test, squared=False)
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))