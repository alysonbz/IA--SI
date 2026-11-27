import numpy as np

# Import Lasso
from sklearn.linear_model import Lasso

# Import train_test_split e KFold
from sklearn.model_selection import train_test_split, KFold

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

from src.utils import load_diabetes_clean_dataset

diabetes_df = load_diabetes_clean_dataset()

X = diabetes_df.drop(['glucose'], axis=1)
y = diabetes_df['glucose'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Inicializar Lasso
lasso = Lasso()

# Inicializar KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definir o conjunto de parâmetros
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

# Inicializar GridSearchCV
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Treinar o regressor
lasso_cv.fit(X_train, y_train)

print("Tuned lasso parameters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))
