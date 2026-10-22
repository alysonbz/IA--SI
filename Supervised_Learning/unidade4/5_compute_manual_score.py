import numpy as np
from src.utils import processing_all_features_sales_clean


def compute_RSS(predictions, y):
    sub_square = np.square(predictions - y)
    RSS = np.sum(sub_square)
    return RSS


def compute_MSE(predictions, y):
    # MSE = RSS / número de observações
    MSE = compute_RSS(predictions, y) / len(y)
    return MSE


def compute_RMSE(predictions, y):
    # RMSE = raiz quadrada do MSE
    RMSE = np.sqrt(compute_MSE(predictions, y))
    return RMSE


def compute_R_squared(predictions, y):
    # R² = 1 - RSS / TSS, onde TSS é a soma dos quadrados totais
    TSS = np.sum(np.square(y - np.mean(y)))
    r_squared = 1 - compute_RSS(predictions, y) / TSS
    return r_squared


X, y, predictions = processing_all_features_sales_clean()

print("RSS: {}".format(compute_RSS(predictions, y)))
print("MSE: {}".format(compute_MSE(predictions, y)))
print("RMSE: {}".format(compute_RMSE(predictions, y)))
print("R^2: {}".format(compute_R_squared(predictions, y)))