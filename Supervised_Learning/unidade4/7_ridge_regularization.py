from src.utils import load_sales_clean_dataset
from sklearn.model_selection import train_test_split
# Importar o KNN para regressão
from sklearn.neighbors import KNeighborsRegressor

# Carregar os dados
sales_df = load_sales_clean_dataset()

# Criar as variáveis X (features) e y (target)
X = sales_df.drop(["sales", "influencer"], axis=1)  # Remover 'sales' e 'influencer' de X
y = sales_df["sales"].values  # 'sales' é a variável alvo

# Dividir os dados em conjunto de treino e teste (sem o parâmetro 'stratify' para variáveis contínuas)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir diferentes valores de K para o KNN (n_neighbors)
k_values = [1, 3, 5, 7, 9, 11]
knn_scores = []

# Variáveis para armazenar o melhor valor de K e a melhor pontuação
best_k = None
best_score = -float('inf')  # Inicializando com o valor mais baixo possível

# Para cada valor de K, criar e treinar o modelo KNN
for k in k_values:
    # Criar o modelo KNN para regressão
    knn = KNeighborsRegressor(n_neighbors=k)

    # Ajustar o modelo nos dados de treino
    knn.fit(X_train, y_train)

    # Obter R-squared no conjunto de teste
    score = knn.score(X_test, y_test)  # R^2 é a métrica de desempenho
    knn_scores.append(score)

    # Verificar se o R-squared atual é o melhor
    if score > best_score:
        best_score = score
        best_k = k

# Exibir todos os resultados
for k, score in zip(k_values, knn_scores):
    print(f'K (n_neighbors): {k}, R-squared: {score}')

# Exibir o melhor K e o melhor R-squared
print(f'\nMelhor K (n_neighbors): {best_k}')
print(f'Melhor R-squared: {best_score}')
