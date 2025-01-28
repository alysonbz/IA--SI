from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Corrigido aqui

# Carregar o conjunto de dados
diabetes_df = load_diabetes_clean_dataset()

# Separar as características e o alvo
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Dividir o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instanciar o modelo
logreg = LogisticRegression()

# Ajustar o modelo
logreg.fit(X_train, y_train)

# Prever as probabilidades
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

# Exibir as primeiras 10 previsões de probabilidade
print(y_pred_probs[:10])