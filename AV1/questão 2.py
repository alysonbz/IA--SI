import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#Carrega o dataset da Questão 1
df = pd.read_csv('hcv_dataset_preparado.csv')

#separa features(X) e target(y)
if 'Target' in df.columns:
    X = df.drop(columns=['Category', 'Target'])
    y = df['Target']
else:
    target_col = 'Category'
    X = df.drop(columns=[target_col])
    y = df[target_col]

#garante apenas colunas numéricas para o cálculo de distâncias do KNN
X = X.select_dtypes(include=[np.number])

#divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

resultados = []

#configuração 1: dados sem normalização ou padronização
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acc_sem = knn.score(X_test, y_test)
resultados.append({"Configuração": 1, "Técnica utilizada": "Sem normalização", "Acurácia": acc_sem})

#configuração 2: técnica 1 (StandardScaler)
scaler_1 = StandardScaler()
X_train_t1 = scaler_1.fit_transform(X_train)
X_test_t1 = scaler_1.transform(X_test)

knn.fit(X_train_t1, y_train)
acc_t1 = knn.score(X_test_t1, y_test)
resultados.append({"Configuração": 2, "Técnica utilizada": "StandardScaler", "Acurácia": acc_t1})

#configuração 3: técnica 2 (MinMaxScaler)
scaler_2 = MinMaxScaler()
X_train_t2 = scaler_2.fit_transform(X_train)
X_test_t2 = scaler_2.transform(X_test)

knn.fit(X_train_t2, y_train)
acc_t2 = knn.score(X_test_t2, y_test)
resultados.append({"Configuração": 3, "Técnica utilizada": "MinMaxScaler", "Acurácia": acc_t2})

#configuração 4: técnica 3 (Normalização Logarítmica)
X_train_t3 = np.log1p(X_train)
X_test_t3 = np.log1p(X_test)

knn.fit(X_train_t3, y_train)
acc_t3 = knn.score(X_test_t3, y_test)
resultados.append({"Configuração": 4, "Técnica utilizada": "Normalização Logarítmica", "Acurácia": acc_t3})

#exibe a tabela final de resultados
df_tabela = pd.DataFrame(resultados)
print("\nTABELA COMPARATIVA DE DESEMPENHO:")
print(df_tabela.to_string(index=False))