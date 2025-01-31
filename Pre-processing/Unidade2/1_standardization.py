from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

wine = load_wine_dataset()

X = wine.drop(['Quality'], axis=1)
y = wine['Quality'].values

# Divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn
knn.fit(X_train, y_train)

# Mostre o acerto do algoritmo
print(knn.score(X_test, y_test))
