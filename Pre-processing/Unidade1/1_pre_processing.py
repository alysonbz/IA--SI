from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd


#VOLUNTEER
print("\nVOLUNTEER")
volunteer = load_volunteer_dataset()
print(volunteer.info())
#HIKING
print("\nHIKING")
hiking = load_hiking_dataset()
print(hiking.head())
print(hiking.info())

#WINE
print("Wine")
wine  = load_wine_dataset()
print(wine.describe())

#DATA FRAME 1
print("\nDATAFRAME 1")
df1 = load_df1_unidade1()

print("\n- Tabela")
print(df1)

print("\n- Excluindo NaN")
print(df1.dropna())

print("\n- Excluindo Linhas")
print(df1.drop([3]))

print("\n- Excluindo Colunas")
print(df1.drop("C", axis=1))

print("\n- Soma NaN")
print(df1.isna().sum())

print("\n- Excluindo NaN das Colunas")
print(df1.dropna(subset=['A']))

#DATA FRAME 2
print("\nDATAFRAME 2")
df2 = load_df2_unidade1()
print("\n- Tabela")
print(df2)

print("\n- Info")
print(df2.info())

print("\n- Converter os Types")
df2["C"] = df2["C"].astype("int64")
print(df2.dtypes)