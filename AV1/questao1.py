import pandas as pd
import numpy as np


df = pd.read_csv("flavors_of_cacao.csv")
dfn = df.dropna()
newdataset = dfn.drop(['Bean\nType'],axis=1)
print(newdataset.isnull().sum())
print("Colunas disponíveis:", list(newdataset.columns))
