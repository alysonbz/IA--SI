import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("sabores_de_cacau_ajustado.csv")

X = df["Cocoa\nPercent"].values
y = df["Rating"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)


#def knn(X_train, y_train, X_test, k, metric):
#    predict = []
#    for test_point in X_test.values:
#        if metric == 'euclidiano':
#            distances = np.sqrt(np.sum((X_train.values - test_point) ** 2, axis=1))
#        elif metric == 'manhattan':
#            distances = np.sum(np.abs(X_train.values - test_point), axis=1)
#        else:
#            raise ValueError("Métrica desconhecida. Escolha 'euclidean' ou 'manhattan'.")
#
#        k_indices = np.argsort(distances)[:k]
#        k_labels = y_train.iloc[k_indices].values
#
#        most_common = Counter(k_labels).most_common(1)[0][0]
#        predict.append(most_common)
#
 #   return predict


#k =
#metric = 'euclidiano'

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

#print("Matriz de confusão:")
#print(confusion_matrix(y_test, y_pred))
#print("\nRelatório de classificação:")
#print(classification_report(y_test, y_pred))
