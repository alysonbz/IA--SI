import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Se o diretório 'src' não estiver no caminho de importação, adicione o seguinte:
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.utils import load_sales_clean_dataset

# Carregar os dados
sales_df = load_sales_clean_dataset()

# Criar arrays X (variáveis independentes) e y (variável dependente)
X = sales_df["tv"].values.reshape(-1, 1)  # "tv" é a variável preditora
y = sales_df["sales"].values  # "sales" é a variável alvo

# Criar o objeto KFold
kf = KFold(n_splits=6, shuffle=True, random_state=5)

# Criar o modelo de regressão linear
reg = LinearRegression()

# Calcular as pontuações de validação cruzada (cross-validation)
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Exibir as pontuações de cada fold
print("Pontuações de cada fold: ", cv_scores)

# Exibir a média das pontuações
print("Média das pontuações de validação cruzada: ", np.mean(cv_scores))

# Exibir o desvio padrão das pontuações
print("Desvio padrão das pontuações de validação cruzada: ", np.std(cv_scores))
