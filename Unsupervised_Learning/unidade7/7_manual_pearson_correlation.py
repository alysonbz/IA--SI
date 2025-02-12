# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    return None


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df.iloc[:, 0].values

# Assign the 1st column of grains: length
length = grains_df.iloc[:, 1].values

# Calculate the Pearson correlation
correlation, _ = pearsonr(width,length)

# Display the correlation
print(correlation)