# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

wine = load_wine_dataset()

# Inicializa o scaler
scaler = StandardScaler()

# Exclua do dataset a coluna Quality
X = wine.drop(['Quality'], axis=1)

# Normalize o dataset com scaler
X_norm = scaler.fit_transform(X)

# Obtenha as labels da coluna Quality
y = wine['Quality'].values

# Print a variância de X
print('Variância:', np.var(X))

# Print a variância do dataset normalizado
print('Variância do dataset normalizado:', np.var(X_norm))

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)

# Inicialize o algoritmo KNN
knn = KNeighborsClassifier()

# Aplique a função fit do KNN
knn.fit(X_train, y_train)

# Verifique o acerto do classificador
print('Score:', knn.score(X_test, y_test))
