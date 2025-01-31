import numpy as np
from src.utils import processing_all_features_sales_clean

# Function to compute RSS
def compute_RSS(predictions, y):
    sub_square = np.square(predictions - y)
    RSS = np.sum(sub_square)
    return RSS

# Function to compute MSE (Mean Squared Error)
def compute_MSE(predictions, y):
    MSE = np.mean(np.square(predictions - y))
    return MSE

# Function to compute RMSE (Root Mean Squared Error)
def compute_RMSE(predictions, y):
    RMSE = np.sqrt(compute_MSE(predictions, y))  # RMSE is the square root of MSE
    return RMSE

# Function to compute R-squared
def compute_R_squared(predictions, y):
    total_sum_of_squares = np.sum(np.square(y - np.mean(y)))  # Total variance in the data
    residual_sum_of_squares = compute_RSS(predictions, y)  # RSS
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)  # R^2 formula
    return r_squared


# Assuming this function processes the data and returns features, true values, and predictions
X, y, predictions = processing_all_features_sales_clean()

# Print the metrics
print("RSS: {}".format(compute_RSS(predictions, y)))
print("MSE: {}".format(compute_MSE(predictions, y)))
print("RMSE: {}".format(compute_RMSE(predictions, y)))
print("R^2: {}".format(compute_R_squared(predictions, y)))
