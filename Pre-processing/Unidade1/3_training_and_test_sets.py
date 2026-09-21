from src.utils import load_volunteer_dataset

from sklearn.model_selection import train_test_split

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(["Latitude", "Longitude"], axis = 1)
print(volunteer_new)
print("____________________________________________________________")

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=["category_desc"])
print(volunteer)
print("____________________________________________________________")

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

print("____________________________________________________________")

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis = 1)
print(X)

print("____________________________________________________________")

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]
print(y)
print("____________________________________________________________")

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

print("____________________________________________________________")

# mostre o balanceamento das classes em 'category_desc' novamente
print(volunteer['category_desc'].value_counts(),'\n','\n')
