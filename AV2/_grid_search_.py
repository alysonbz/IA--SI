import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from _preparacao_ import *

valores_alpha = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# GRID SEARCH RIDGE
grid_ridge = GridSearchCV(
    estimator=Ridge(),
    param_grid={'alpha': valores_alpha},
    scoring='neg_mean_squared_error',
    cv=kfold
)

grid_ridge.fit(X_train_scaled_df, y_train)

resultados_ridge = pd.DataFrame({
    'Alpha': valores_alpha,
    'MSE médio': -grid_ridge.cv_results_['mean_test_score']
})

# GRID SEARCH LASSO
grid_lasso = GridSearchCV(
    estimator=Lasso(max_iter=100000),
    param_grid={'alpha': valores_alpha},
    scoring='neg_mean_squared_error',
    cv=kfold
)

grid_lasso.fit(X_train_scaled_df, y_train)

resultados_lasso = pd.DataFrame({
    'Alpha': valores_alpha,
    'MSE médio': -grid_lasso.cv_results_['mean_test_score']
})

# MELHORES MODELOS
melhor_ridge = grid_ridge.best_estimator_
melhor_lasso = grid_lasso.best_estimator_


if __name__ == "__main__":
    print("--- RESULTADOS GRID SEARCH RIDGE ---")
    print(resultados_ridge)

    print("\nMelhor configuração Ridge:")
    print(grid_ridge.best_params_)
    print("Menor MSE médio:", -grid_ridge.best_score_)

    print("\n--- RESULTADOS GRID SEARCH LASSO ---")
    print(resultados_lasso)

    print("\nMelhor configuração Lasso:")
    print(grid_lasso.best_params_)
    print("Menor MSE médio:", -grid_lasso.best_score_)