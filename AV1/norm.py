from escalas import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_treino_final, y_treino)

y_pred = knn.predict(X_teste_final)

acuracia = accuracy_score(y_teste, y_pred)

print("Acurácia com os dados padronizados:", acuracia)