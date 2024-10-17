
from src.utils import load_hiking_dataset
from sklearn.preprocessing import LabelEncoder

hiking = load_hiking_dataset()


# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking["Accessible"])

# Compare the two columns
print(hiking[['Accessible_enc', "Accessible"]].head())

