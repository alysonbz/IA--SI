from src.utils import load_volunteer_dataset
<<<<<<< HEAD
from sklearn.model_selection import train_test_split
=======
_____
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
<<<<<<< HEAD
volunteer_new = volunteer.drop(['Latitude', 'Longitude'],axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset='category_desc')

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train.value_counts())
=======
volunteer_new = __

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = ___

# mostre o balanceamento das classes em 'category_desc'
print(___['category_desc'].__,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.__(__, axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = __[['__']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
___
>>>>>>> 8dbee5f0bdad0e083bc03654e1a4101bf868fd0d
