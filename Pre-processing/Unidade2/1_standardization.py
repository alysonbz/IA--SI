from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

wine = load_wine_dataset()

X = wine.drop(['Quality'],axis=1)

y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

# Aplique a função fit do knn
<<<<<<< HEAD
knn.fit(X_train, y_train)
=======
knn.KNeighbors(X_train, y_train)
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

# mostre o acerto do algoritmo
print(knn.score(X_test, y_test))