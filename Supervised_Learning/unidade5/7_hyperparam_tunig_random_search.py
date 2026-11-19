import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

logreg = LogisticRegression(solver="liblinear")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    "penalty": ["l1", "l2"],
    "tol": np.linspace(0.0001, 0.01, 50),
    "C": np.linspace(0.0001, 1, 50),
    "class_weight": ["balanced", {0: 0.2, 1: 0.8}],
}

logreg_cv = RandomizedSearchCV(
    logreg, params, cv=kf, n_iter=20, random_state=42
)

logreg_cv.fit(X_train, y_train)

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))