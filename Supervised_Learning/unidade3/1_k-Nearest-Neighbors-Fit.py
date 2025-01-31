from src.utils import load_churn_dataset
import numpy as np

# Importando o KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

churn_df = load_churn_dataset()

# Criando arrays para as variáveis de entrada e saída
y = churn_df["Churn"].values
X = churn_df[["Age", "Balance"]].values

# Criando um classificador KNN com 6 vizinhos
knn = KNeighborsClassifier(n_neighbors=6)

# Treinando o classificador com os dados
knn.fit(X, y)

X_test = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# Predizendo as labels para os dados de teste
y_pred = knn.predict(X_test)

# Imprimindo as previsões para X_test
print("Previsões: {}".format(y_pred))
