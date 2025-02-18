import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Dataset atualizado
classificacao = pd.read_csv('./dataset/dataset_classificacao_ajustado.csv')

classificacao_df = classificacao
X = classificacao_df.drop(['booking_status'],axis=1)
y = classificacao_df['booking_status'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reduzir para no máximo 5000 amostras para manter a comparabilidade com AV1
classificacao_df = classificacao_df.sample(n=5000, random_state=42)

# Inicializar o Knn
knn = KNeighborsClassifier()

# Incializar o Kfold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Calcular a matriz de covariância para Mahalanobis
cov = EmpiricalCovariance().fit(X_train)
VI = np.linalg.inv(cov.covariance_)  # Matriz inversa da covariância

# Criar espaço de busca dos hiperparâmetros
params = [
    {'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['chebyshev', 'manhattan', 'euclidean']},
    {'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['mahalanobis'], 'metric_params': [{'VI': VI}]}
]

# Criar e executar GridSearchCV
knn_cv = GridSearchCV(knn, params, cv=kf)

# Ajustar os dados ao modelo
knn_cv.fit(X_train, y_train)

# Obter os melhores parâmetros
melhores_parametros = knn_cv.best_params_

# Omitir a matriz VI da exibição
if 'metric_params' in melhores_parametros and 'VI' in melhores_parametros['metric_params']:
    melhores_parametros['metric_params'] = {'VI': 'Matriz omitida'}

# Exibir os melhores parâmetros sem a matriz VI
print("Melhores parâmetros encontrados:", melhores_parametros)
print("Melhor acurácia obtida:", knn_cv.best_score_)

# Criando um DataFrame com os resultados completos
results_df = pd.DataFrame(knn_cv.cv_results_)[['param_metric', 'param_n_neighbors', 'mean_test_score']]

# Obtendo o melhor k para cada métrica
best_k_per_metric = results_df.loc[results_df.groupby('param_metric')['mean_test_score'].idxmax()]
print("\nMelhor 'k' para cada métrica:")
print(best_k_per_metric)