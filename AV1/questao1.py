# Carregar o dataset
from src.utils import load_drug_dataset

drug = load_drug_dataset()

if drug.isnull().sum().sum() > 0:
    print("Existem valores NaN. Removendo...")
    drug_cleaned = drug.dropna()
else:
    print("Não há valores NaN.")
    drug_cleaned = drug.copy()

print("Colunas disponíveis:", drug_cleaned.columns)

columns_to_keep = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
drug_filtered = drug_cleaned[columns_to_keep]

print("\nDistribuição de classes:")
print(drug_filtered['Drug'].value_counts())

drug_filtered['Drug'] = drug_filtered['Drug'].astype('category').cat.codes
print("\nClasses convertidas para valores numéricos.")


# Exibir o dataframe final
print("\nDataframe final:\n", drug_filtered.head())

# Salvar o dataset atualizado
drug_filtered.to_csv("drug_dataset_ajustado.csv", index=False)
print("\nDataset ajustado salvo como 'drug_dataset_ajustado.csv'.")