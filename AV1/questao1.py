from src.utills import diabetes_dataset

diabetes = diabetes_dataset()

print('Verificando se a células vazias')
print(diabetes.isnull().sum())

if diabetes.isnull().sum().sum() > 0:
    diabetes = diabetes.dropna()
    print("Células vazias removidas.")

colunas_relevantes = ['Pregnancies', 'Glucose', 'BloodPressure',
                      'SkinThickness', 'Insulin', 'BMI',
                      'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_relevante = diabetes[colunas_relevantes]

diabetes_relevante.rename(columns={'Outcome': 'Class'}, inplace=True)

print("\nDataframe final:")
print(diabetes_relevante.head())

print("\nDistribuição da classe 'Class':")
print(diabetes_relevante['Class'].value_counts())

output_path = "diabetes_ajustado.csv"
diabetes_relevante.to_csv(output_path, index=False)
print(f"\nDataset salvo como {output_path}.")