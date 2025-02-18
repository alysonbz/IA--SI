import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib

# Carregar os dados
df = pd.read_csv('sabores_de_cacau_ajustado.csv', names=['company', 'specificorigin', 'cocoapercent', 'location', 'rating', 'beantype', 'broadorigin'], header=0)
nonnum = ['company', 'specificorigin', 'location', 'beantype', 'broadorigin']

def onehot_encode(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)

# Aplicar one-hot encoding
df = onehot_encode(df, [col for col in nonnum if col in df.columns])

df.to_csv("sabores_de_cacau_ajustado.csv", index=False)

# Separação de variáveis independentes e dependentes
y = df['rating']
X = df.drop('rating', axis=1)

# Normalização dos dados
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Definição do modelo KNN
knn = KNeighborsClassifier()

# Definição dos parâmetros para GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

n_cores = joblib.cpu_count()

# Aplicação do GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=n_cores)
grid_search.fit(X_train, y_train)

print("Melhor combinação de parâmetros:", grid_search.best_params_)

# Previsões
y_pred = grid_search.best_estimator_.predict(X_test)

# Cálculo da acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo com os melhores parâmetros:", accuracy)
