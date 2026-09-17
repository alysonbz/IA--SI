from src.utils import load_volunteer_dataset
from sklearn.model_selection import train_test_split

volunteer = load_volunteer_dataset()

print("\n* --># Exclua as colunas Latitude e Longitude de volunteer:\n")
volunteer_new = volunteer.drop(["Latitude","Longitude"],axis=1)
print(volunteer_new)

print("\n* --># Exclua as linhas com valores null da coluna category_desc de volunteer_new:\n")
volunteer = volunteer_new.dropna(subset= ["category_desc"])
print(volunteer)

print("\n* --># mostre o balanceamento das classes em 'category_desc':\n")
print(volunteer['category_desc'].value_counts(),'\n','\n')

print("\n* --># Crie um DataFrame com todas as colunas, com exceção de ``category_desc``:\n")
X = volunteer.drop(['category_desc'], axis=1)
print(X)

print("\n* --># Crie um dataframe de labels com a coluna category_desc:\n")
y = volunteer[['category_desc']]
print(y)

print("\n* --># # Utiliza a a amostragem stratificada para separar o dataset em treino e teste:\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

print("\n* --># mostre o balanceamento das classes em 'category_desc' novamente:\n")
print(volunteer['category_desc'].value_counts(),'\n','\n')