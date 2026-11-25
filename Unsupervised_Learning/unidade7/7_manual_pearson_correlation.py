# Perform the necessary imports
import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_grains_dataset


def pearson_correlation(x, y):
    # Calcula as médias de x e y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calcula o numerador (covariância)
    numerator = np.sum((x - mean_x) * (y - mean_y))

    # Calcula o denominador (produto dos desvios)
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

    return numerator / denominator if denominator != 0 else 0.0


grains_df = load_grains_dataset()

# Assign the 0th column of grains: width
width = grains_df.iloc[:, 0].values

# Assign the 1st column of grains: length
length = grains_df.iloc[:, 1].values

# Calculate the Pearson correlation
correlation = pearson_correlation(width, length)

# Display the correlation
print(correlation)
