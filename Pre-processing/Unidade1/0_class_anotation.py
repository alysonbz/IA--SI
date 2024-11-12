from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
print(volunteer.info())

hiking = load_hiking_dataset()
print(hiking.info())

wine  = load_wine_dataset()
print(wine.describe())

df1 = load_df1_unidade1()
print(df1)
print(df1.dropna())
print(df1.dropna([1,2,3]))
print()

df2 = load_df2_unidade1()







