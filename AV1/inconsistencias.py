from dados import *

heart_limpo = heart.dropna()

X_limpo = heart_limpo.drop(columns="num")

y_limpo = (heart_limpo["num"] > 0).astype(int)

variaveis_numericas = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

if __name__ == "__main__":
    print("\nVerificando elementos nulos:")
    print(heart.isna().sum())

    print("\nQuantidade de linhas duplicadas:")
    print(heart.duplicated().sum())

    print("\nVerificação de outliers:")

    for coluna in variaveis_numericas:
        Q1 = heart_limpo[coluna].quantile(0.25)
        Q3 = heart_limpo[coluna].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        quantidade = (
                (heart_limpo[coluna] < limite_inferior) |
                (heart_limpo[coluna] > limite_superior)
        ).sum()

        print(f"{coluna}: {quantidade} outliers")
