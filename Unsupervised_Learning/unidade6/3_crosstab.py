import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

# Carregar o dataset
samples_df = load_grains_dataset()

# Separar as amostras e os rótulos
samples = samples_df.drop(['variety', 'variety_number'], axis=1)
varieties = samples_df['variety'].values

# Criar um modelo KMeans com 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# Ajustar o modelo e obter os rótulos dos clusters
labels = model.fit_predict(samples)

# Criar um DataFrame com os rótulos dos clusters e as variedades reais
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Criar a tabela cruzada
ct = pd.crosstab(df['labels'], df['varieties'])

# Exibir a tabela cruzada
print(ct)

