import numpy as np
from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression


class KFold:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _compute_score(self, obj, X_train, y_train, X_test, y_test):
        # Treina o modelo com os dados de treino da dobra atual e calcula o score no teste
        obj.fit(X_train, y_train)
        return obj.score(X_test, y_test)

    def cross_val_score(self, obj, X, y):
        scores = []

        # parte 1: dividir o dataset X e y em n_splits vezes
        X_folds = np.array_split(X, self.n_splits)
        y_folds = np.array_split(y, self.n_splits)

        # parte 2: Calcular a métrica score para cada subset dividido na parte 1
        for i in range(self.n_splits):
            # O fold atual (i) é separado para teste/validação
            X_test = X_folds[i]
            y_test = y_folds[i]

            # Todos os outros folds são concatenados para formar o conjunto de treino
            X_train = np.concatenate([X_folds[j] for j in range(self.n_splits) if j != i])
            y_train = np.concatenate([y_folds[j] for j in range(self.n_splits) if j != i])

            # Chamar a função _compute_score para cada subset
            score = self._compute_score(obj, X_train, y_train, X_test, y_test)

            # appendar na lista scores cada valor obtido na parte 2
            scores.append(score)

        # parte 3 - retornar a lista de scores
        return scores


sales_df = load_sales_clean_dataset()

# Create X and y arrays
X = sales_df["tv"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Create a KFold object
kf = KFold(n_splits=6)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = kf.cross_val_score(reg, X, y)

# Print scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))

