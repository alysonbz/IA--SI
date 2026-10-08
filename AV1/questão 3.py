import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#carregar o dataset preparado na questão 1
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

#divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#aplicar a técnica de normalização escolhida na questão 2 (Normalização Logarítmica)
X_train_norm = np.log1p(X_train)
X_test_norm = np.log1p(X_test)

#avaliar diferentes valores de K (15 valores distintos variando de 1 a 15)
neighbors = np.arange(1, 16)
train_accuracies = {}
test_accuracies = {}
resultados_k = []

for neighbor in neighbors:
    #inicializa o KNN com o número iterativo de vizinhos
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train_norm, y_train)

    #calcula as acurácias de treino e teste
    acc_train = knn.score(X_train_norm, y_train)
    acc_test = knn.score(X_test_norm, y_test)

    train_accuracies[neighbor] = acc_train
    test_accuracies[neighbor] = acc_test

    resultados_k.append({
        "Valor de K": neighbor,
        "Acurácia de treinamento": acc_train,
        "Acurácia de validação": acc_test
    })

#tabela de resultados
df_tabela_k = pd.DataFrame(resultados_k)
print("\nTABELA COMPARATIVA: VALOR DE K VS ACURÁCIA")
print(df_tabela_k.to_string(index=False))

#gráfico de linhas
plt.figure(figsize=(9, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, list(train_accuracies.values()), label="Training Accuracy", marker='o', color='blue')
plt.plot(neighbors, list(test_accuracies.values()), label="Testing Accuracy", marker='s', color='red')
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig('grafico_knn.png', dpi=300, bbox_inches='tight')

