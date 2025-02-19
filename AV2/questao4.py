import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Carregar o dataset
data = pd.read_csv("drug200.csv")

# Pré-processamento dos dados
# Codificar variáveis categóricas
label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separar as features e o target
X = data.drop(columns=['Drug'])
y = data['Drug']

# Padronizar as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means com k=4
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar os rótulos dos clusters ao DataFrame original
data['Cluster'] = clusters

# Criar a tabela cruzada (crosstab)
crosstab = pd.crosstab(data['Cluster'], data['Drug'])

# Exibir a tabela cruzada
print(crosstab)