import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carregar os dados (substitua pelos seus dados X e y)
data = load_iris()
X = data.data
y = data.target

# 2. Padronizar os dados (média = 0, desvio padrão = 1)
# Passo essencial porque o PCA é sensível à escala das variáveis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Instanciar e aplicar o PCA reduzindo para 2 dimensões
# Dica: você também pode passar um float como n_components=0.95 para manter 95% da variância
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Verificar a redução de dimensões e a informação preservada
print(f"Formato original: {X.shape}")
print(f"Formato reduzido: {X_pca.shape}")
print(f"Variância explicada por componente: {pca.explained_variance_ratio_}")
print(f"Variância total preservada: {sum(pca.explained_variance_ratio_):.2%}")

# 4. Visualizar o resultado em 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=60)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)')
plt.title('Redução de Dimensionalidade com PCA')
plt.colorbar(label='Classes')
plt.show()
