# Import StandardScaler
from tkinter import Scale

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


wine = load_wine_dataset()

# Inicializer o scale
scaler = ___

# exclua do dataset a coluna Quality
X = wine.drop(wine['Quality'],axis=1)

#normalize o dataset com scaler
X_norm = scaler.fit_transform(X)

#obtenha as labels da coluna Quality
y = wine['Quality'].values(





)

#print a valriância de X
print('variancia',__)

#print a variânca do dataset X_norm
print('variancia do dataset normalizado',__)

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)

#inicialize o algoritmo KNN
knn = KNeighborsClassifier()

# Aplique a função fit do KNN
knn.__(__,__)

# Verifique o acerto do classificador
print('score', knn.__(__, __))