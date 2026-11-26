import numpy as np
from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Create X from the radio column's values
X = sales_df ("radio", axis=1).values

# Create y from the sales column's values
y = sales_df ["radio"].values

# Reshape X
x_bm1 = X[:,3]
x_bm1 = x_bm1.reshape(-1, 1)

# Check the shape of the features and targets
print(type(X), type(y))