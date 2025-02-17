import matplotlib.pyplot as plt
from src.utils import load_sales_clean_dataset

# Import Lasso
<<<<<<< HEAD
from sklearn.linear_model import Lasso
=======
from sklearn.linear import Lasso
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d

sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df.drop(["sales","influencer"], axis=1)
y = sales_df["sales"].values
sales_columns = X.columns

# Instantiate a lasso regression model
<<<<<<< HEAD
lasso = Lasso(alpha = 0.3)

# Compute and print the coefficients
lasso_coef = lasso.fit(X,y).coef_
=======
lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
lasso_coef = Lasso.fit(X,y,)
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()