from src.utils import load_sales_clean_dataset
# Import Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sales_df = load_sales_clean_dataset()
# Create X and y arrays
X = sales_df.drop(["sales", "influencer"], axis=1)
y = sales_df["sales"].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []

for alpha in alphas:
    # Create a Ridge regression model
    ridge = Ridge(alpha=alpha)

    # Fit the data
    ridge.fit(X_train, y_train)

    # Obtain R-squared
    score = ridge.score(X_test, y_test)  # Corrected the method to `score`
    ridge_scores.append(score)

# Print the ridge_scores
print(ridge_scores)
