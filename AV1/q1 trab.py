import pandas as pd
vinhored= pd.read_csv("datasetwinw/winequality-red.csv", sep=";")
vinhowhite=pd.read_csv("datasetwinw/winequality-white.csv", sep =";")
# sem o "sep" é criado um dataset de uma coluna só por não saber como separar
print(vinhored)
print(vinhowhite)