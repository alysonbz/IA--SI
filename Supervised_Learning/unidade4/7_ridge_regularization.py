from src.utils import load_sales_clean_dataset
# Import Ridge
<<<<<<< HEAD
from sklearn.linear_model import Ridge
=======
from scikit.learn import Ridge
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d


sales_df = load_sales_clean_dataset()
# Create X and y arrays
X = sales_df.drop(["sales","influencer"], axis=1)
y = sales_df["sales"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:
    # Create a Ridge regression model
    ridge = Ridge(alpha=alpha)

    # Fit the data
<<<<<<< HEAD
    ridge.fit(X_train, y_train)

    # Obtain R-squared
    score = ridge.score(X_test, y_test)
=======
    ridge.fit = (X_train, y_train)

    # Obtain R-squared
    score = ridge.scores(X_test,y_test)
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d
    ridge_scores.append(score)
print(ridge_scores)