import numpy as np

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import KFold
from sklearn.model_selection import KFold

# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

from src.utils import load_diabetes_clean_dataset

# Load dataset
diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Logistic Regression
logreg = LogisticRegression(solver='liblinear', max_iter=1000)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create the parameter space
params = {
    "neighbour": [8,5,4,9,7,10],  # Regularization type
   "class_weight": ["balanced", {0: 0.6, 1: 0.4}]  # Class weights
}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, param_distributions=params, cv=kf, n_iter=20, random_state=42, n_jobs=-1)

# Fit the data to the model
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {:.4f}".format(logreg_cv.best_score_))
