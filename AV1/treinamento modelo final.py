import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#carregar o dataset preparado na Questão 1
df = pd.read_csv('hcv_dataset_preparado.csv')

#separar features(X) e target(y)
if 'Target' in df.columns:
    X = df.drop(columns=['Category', 'Target'])
    y = df['Target']
else:
    target_col = 'Category'
    X = df.drop(columns=[target_col])
    y = df[target_col]

X = X.select_dtypes(include=[np.number])

#treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#aplicar a normalização Logarítmica
X_train_norm = np.log1p(X_train)
X_test_norm = np.log1p(X_test)

#treinamento do Modelo Final
k_otimo = 5
knn_final = KNeighborsClassifier(n_neighbors=k_otimo)
knn_final.fit(X_train_norm, y_train)

#teste
acc_teste_final = knn_final.score(X_test_norm, y_test)

print("AVALIAÇÃO DO MODELO FINAL")
print("Valor de K utilizado:")
print(k_otimo)
print("\nacurácia final no conjunto de teste:")
print(acc_teste_final)