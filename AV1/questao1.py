import pandas as pd

df = pd.read_csv("flavors_of_cacao.csv")
print("Resumo:")
print(df.info())

df = df.dropna()

df = df.drop(["REF", "Review\nDate"], axis=1)

print(df.dtypes)

def percentfloat(df):
    return df.apply(lambda x: float(x.strip('%'))/100)

df['Cocoa\nPercent'] = percentfloat(df["Cocoa\nPercent"])

print(df.dtypes)

df.to_csv("sabores_de_cacau_ajustado.csv", index=False)