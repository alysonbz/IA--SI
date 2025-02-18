import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset atualizado
classificacao = pd.read_csv('./dataset/dataset_classificacao_ajustado.csv')

# Selecionar os dois atributos mais relevantes
top_2_features = ["lead_time", "no_of_special_requests"]

# Criar um subconjunto do dataset com os atributos selecionados
X_selected = classificacao[top_2_features]

# Definir o melhor K
best_k_silhouette = 2

# Aplicar KMeans com o K obtido pelo índice de silhueta
kmeans = KMeans(n_clusters=best_k_silhouette, random_state=42)
classificacao['Cluster'] = kmeans.fit_predict(X_selected)

# Criar um DataFrame para crosstab com os clusters e a variável alvo
df = pd.DataFrame({'Cluster': classificacao['Cluster'], 'Booking_Status': classificacao['booking_status']})

# Criar crosstab para verificar distribuição dos clusters em relação à variável alvo
crosstab_result = pd.crosstab(df['Cluster'], df['Booking_Status'])

# Exibir a tabela crosstab
print("Distribuição dos clusters em relação à variável alvo:")
print(crosstab_result)

# Criar um gráfico de dispersão (scatterplot) para visualizar os clusters e a variável alvo
plt.figure(figsize=(8, 6))
plt.scatter(X_selected['lead_time'], X_selected['no_of_special_requests'], c=classificacao['Cluster'], cmap='viridis', s=50, alpha=0.6)
plt.title(f'Distribuição dos Clusters (K={best_k_silhouette})')
plt.xlabel('Tempo de Antecedência')
plt.ylabel('Número de Pedidos Especiais')
plt.colorbar(label='Cluster')
plt.show()