import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Carregar o dataset
dataset_path = 'Dataset_coletado.csv' # Substitua pelo caminho correto
df = pd.read_csv(dataset_path)

