from src.utils import load_sales_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt

sales_df = load_sales_clean_dataset()

y = sales_df["sales"].values
X = sales_df["tv"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import LinearRegression
from sklearn.linear_model import LinearRegression 
# Create the model
reg =  LinearRegression()

# Fit the model to the data
reg.fit(X_train,y_train)

# Make predictions
y_pred = reg.predict(X_test)

print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_pred, y_test, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

y_pred = reg.predict(X_test)

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X_test,y_pred, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()

