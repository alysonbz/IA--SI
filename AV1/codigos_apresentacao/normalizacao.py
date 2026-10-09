from escalas import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def acc(num_treino, num_teste):
    Xtr = pd.concat([num_treino, X_treino_categorico], axis=1)
    Xte = pd.concat([num_teste, X_teste_categorico], axis=1)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(Xtr, y_treino)
    return accuracy_score(y_teste, knn.predict(Xte))

# Não normalizado
tr = X_treino[variaveis_numericas]
te = X_teste[variaveis_numericas]
print("Sem normalizar:", acc(tr, te))

# Min-Max
mm = MinMaxScaler()
tr_mm = pd.DataFrame(mm.fit_transform(tr), columns=variaveis_numericas, index=tr.index)
te_mm = pd.DataFrame(mm.transform(te), columns=variaveis_numericas, index=te.index)
print("Min-Max:", acc(tr_mm, te_mm))

# Log
print("Log:", acc(np.log1p(tr), np.log1p(te)))

# Z-score
print("Z-score:", acc(X_treino_numerico, X_teste_numerico))

resultados = pd.DataFrame({
    "Configuração": [1, 2, 3, 4],
    "Técnica utilizada": ["Sem normalização", "Z-score", "Min-Max", "Log (log1p)"],
    "Acurácia": [
        acc(tr, te),
        acc(X_treino_numerico, X_teste_numerico),
        acc(tr_mm, te_mm),
        acc(np.log1p(tr), np.log1p(te))
    ]
})
print(resultados)
