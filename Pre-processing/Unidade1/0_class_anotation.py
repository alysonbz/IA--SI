from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd


volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


print(hiking.head(3)) #mostra as 5 primeiras linhas e 11 colunas de uma tabela sobre lugares, mostrando identificação, nome... (se a gente quiser, pode mostrar a quantidade que a gente quiser)
print(hiking.info()) #mostra as informações gerais, de uma coluna com algumas variáveis, o tipo que essas variáveis são. (Caracteristicas do dataset)
print(wine.describe())#mostra a descrição dos vinhos, a qualidade dos vinhos,  teor de alcool e o quanto dilui. (dados estatísticos)
#A principal diferença é que o head mostra as linhas principais, como se fosse o cabeçalho
#A info vai mostrar as variáveis que estão nessa tabela, como por exemplo o ID, nome, local
#O describe, vai descrever o que estão nas variáveis,
print(wine.info())
print(df1)
print(df1.dropna()) #tira os numero nulos
print(df1.drop([1,4]))#os numeros em parenteses sao retirados do dataset, nessa situação vai mostrar apenas os valos 0,2,3.
print(df1.drop("A", axis=1))
print(df1.isna().sum()) #dropa os elementos nulos, quantos nulos em cada linha
print(df1.dropna(subset=["B"])) #vai dropar os elementos nulos na coluna desejada, por exemplo na B
print(df1.dropna(thresh =2)) #so vai dropar quando tiver a quantidade de elementos nulos que vc msm define. por exemplo 2

