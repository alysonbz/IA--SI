from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Remove as linhas com valores NaN em qualquer coluna
volunteer_cleaned = volunteer.dropna()

# Exibe o shape do dataset após a remoção de valores NaN
print("Shape do dataset limpo:", volunteer_cleaned.shape)

# Exibe os primeiros dados do dataset limpo para conferir
print(volunteer_cleaned.head())
