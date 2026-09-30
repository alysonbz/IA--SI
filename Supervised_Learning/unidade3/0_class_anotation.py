from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

churn_df = load_churn_dataset()

# Visualizar as primeiras linhas do dataset
print(churn_df.head())

# Verificar os tipos de dados e valores nulos
print(churn_df.info())

# Verificar a distribuição do alvo (target) de churn
print(churn_df['churn'].value_counts())


