from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

wine = load_wine_dataset()

#Coluna alvo
X = wine.drop(['Quality'],axis=1)
X =np.log(X)

y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn
knn.fit(X_train, y_train)

pred=knn.predict(X_test)

# mostre o acerto do algoritmo
print(knn.score(X_test, y_test))

print("knn result",pred, "label:", y_test)