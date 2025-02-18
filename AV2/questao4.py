import pandas as pd
from sklearn.cluster import KMeans

# Carregar os dados
df = pd.read_csv('sabores_de_cacau_ajustado.csv')

# Separar features e rótulos
samples = df.drop(columns=['variety', 'variety_number'])
varieties = df['variety'].values

# Criar um modelo KMeans com 2 clusters
model = KMeans(n_clusters=2, random_state=42, n_init=10)

# Ajustar o modelo e obter rótulos dos clusters
labels = model.fit_predict(samples)

# Criar DataFrame com os rótulos e variedades
df_clusters = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Criar crosstab
ct = pd.crosstab(df_clusters['labels'], df_clusters['varieties'])

# Exibir c
print(ct)