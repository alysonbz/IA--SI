import numpy as np
from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Criação de X com os valores da coluna "radio"
X = sales_df["radio"].values

# Criação de y com os valores da coluna "sales"
y = sales_df["sales"].values

# Reajuste de X
X = X.reshape(-1, 1)

# Verifique a forma das variáveis X e y
print(X.shape, y.shape)
