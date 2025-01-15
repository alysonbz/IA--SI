import matplotlib.pyplot as plt
from src.utils import load_sales_clean_dataset

# Import Lasso
<<<<<<< HEAD
from sklearn.linear_model import Lasso
=======
from sklearn.linear import Lasso
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df.drop(["sales","influencer"], axis=1)
y = sales_df["sales"].values
sales_columns = X.columns

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
<<<<<<< HEAD
lasso_coef = lasso.fit(X, y).coef_
=======
lasso_coef = Lasso.fit(X,y,)
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()