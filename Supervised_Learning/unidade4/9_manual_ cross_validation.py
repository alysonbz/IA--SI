import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


class KFoldCustom:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, model, X_train, y_train, X_val, y_val):
        # Fit the model and compute R-squared or any other score
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)  # Returns R-squared for regression tasks

    def cross_val_score(self, model, X, y):
        scores = []
        # Create KFold object from sklearn to split the data
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Loop through each split
        for train_index, val_index in kf.split(X):
            # Split the data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Compute the score for this fold and append to the scores list
            score = self._compute_score(model, X_train, y_train, X_val, y_val)
            scores.append(score)

        return scores


sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFoldCustom object
kf = KFoldCustom(n_splits=6)

# Create a LinearRegression model
reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = kf.cross_val_score(reg, X, y)

# Print scores
print("Cross-validation scores:", cv_scores)

# Print the mean
print("Mean score:", np.mean(cv_scores))

# Print the standard deviation
print("Standard deviation:", np.std(cv_scores))
