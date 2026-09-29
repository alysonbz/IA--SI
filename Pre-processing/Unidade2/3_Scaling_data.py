# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from src.utils import load_wine_dataset

#df_scaled = pd.DataFrame(scaler.fit_transform(df),columns = df.columns)

wine = load_wine_dataset()

print("\n* --># Inicializer o scale:\n")
scaler = StandardScaler()

print("\n* --># exclua do dataset a coluna:\n")
X = wine.drop(['Quality'],axis=1)

print("\n* -->#normalize o dataset com scaler:\n")
X_norm = scaler.fit_transform(X)

print("\n* -->#obtenha as labels da coluna Quality:\n")
y = wine['Quality'].var

print("\n* -->#print a valriância de X:\n")
print('variancia',X)

print("\n* -->#print a variânca do dataset X_norm:\n")
print('variancia do dataset normalizado',X_norm)


#ATE AQUI FUNCIONA ... ABAIXO AINDA NAO!

print("\n* --># Divida o dataset em treino e teste com amostragem estratificada:\n")
X_train, X_test, y_train, y_test = train_test_split(X_norm, X, y, random_state=42)

print("\n* -->#inicialize o algoritmo KNN:\n")
knn = KNeighborsClassifier

print("\n* --># Aplique a função fit do KNN:\n")
knn.fit(X_train,y_train)

print("\n* --># Verifique o acerto do classificador:\n")
print('score', knn.fit(X_train, y_train))