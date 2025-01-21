import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions, y):
    RSS = np.sum(np.square(predictions - y))
    return RSS

def compute_MSE(predictions, y):
    MSE = np.mean(np.square(predictions - y))
    return MSE

def compute_RMSE(predictions, y):
    RMSE = np.sqrt(compute_MSE(predictions, y))
    return RMSE

def compute_R_squared(predictions, y):
    total = np.sum(np.square(y - np.mean(y)))
    residual = compute_RSS(predictions, y)
    r_squared = 1 - (residual / total)
    return r_squared


X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))