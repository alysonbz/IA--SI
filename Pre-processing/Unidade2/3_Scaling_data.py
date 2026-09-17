# Import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from utils import load_wine_dataset

wine = load_wine_dataset()

# Inicializer o scale
scaler = StandardScaler()

# exclua do dataset a coluna
X = wine.drop(['Quality'], axis=1)

#normalize o dataset com scaler
X_norm = scaler.fit_transform(X)

#obtenha as labels da coluna Quality
y = wine['Quality'].values

#print a variância de X
print('variancia', X.var())

#print a variânca do dataset X_norm
print('variancia do dataset normalizado', X_norm.var())

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split( X, y, stratify= y)

#inicialize o algoritmo KNN
knn = KNeighborsClassifier()


# Aplique a função fit do KNN
knn.fit(X_train, y_train)

# Verifique o acerto do classificador
print('score', knn.score(X_test, y_test))