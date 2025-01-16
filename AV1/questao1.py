import pandas as pd

def load_customer_data():
    return pd.read_csv('customer_data.csv')

customer_data = load_customer_data()

def load_payment_data():
    return pd.read_csv('payment_data.csv')

payment_data = load_payment_data()

print(payment_data.head())
print(customer_data.head())

payment_data_cols = payment_data.drop(['update_date', 'report_date'], axis=1)

#Verificar valores ausentes
print(payment_data_cols.isnull().sum())
print(customer_data.isnull().sum())

#Tratar valores ausentes
payment_data_cols.dropna(inplace=True)
customer_data.dropna(inplace=True)

#Remover duplicatas
payment_data_cols.drop_duplicates(inplace=True)
customer_data.drop_duplicates(inplace=True)

#Verificar valores ausentes
print(payment_data_cols.isnull().sum())
print(customer_data.isnull().sum())

# Combinar os dois DataFrames usando a coluna 'id'
combined_data = pd.merge(payment_data_cols, customer_data, on='id')

#Features e Target
X = combined_data.drop(['label'], axis=1)  # Features
y = combined_data['label']  # Target

#Dimensões dos dados processados
print("Dimensões dos dados de features:", X.shape)
print("Dimensões dos dados de rótulos:", y.shape)

print(combined_data.head())  #Print do DataFrame final
print("Distribuição de classes:")
print(y.value_counts())  #Mostra a distribuição de classes

combined_data.to_csv('credit_risk_data.csv', index=False)
print("Dataset alterado salvo como 'credit_risk_data.csv'")