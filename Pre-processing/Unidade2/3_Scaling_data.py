# Import StandardScaler
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler
=======
from sklearn.preprocessing import ____, StandardScaler
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


wine = load_wine_dataset()

# Inicializer o scale
scaler = StandardScaler()

# exclua do dataset a coluna Quality
X = wine.drop(['Quality'],axis=1)

#normalize o dataset com scaler
X_norm = scaler.fit_transform(X)
<<<<<<< HEAD

=======
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1
#obtenha as labels da coluna Quality
y = wine['Quality'].values

#print a valriância de X
<<<<<<< HEAD
print('variancia',X.var())

#print a variânca do dataset X_norm
print('variancia do dataset normalizado',X.var(X_norm))
=======
print('variancia',X.var)

#print a variânca do dataset X_norm
print('variancia do dataset normalizado',X_norm.var)
>>>>>>> 1239a00c96cd4d3adea696c64633d52b04d5adf1

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)

#inicialize o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Aplique a função fit do KNN
knn.fit(X_train,y_train)

# Verifique o acerto do classificador
print('score', knn.score(X_test, y_test))