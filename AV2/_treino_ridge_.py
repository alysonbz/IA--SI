import pandas as pd
from sklearn.linear_model import Ridge
from _preparacao_ import *

valores_alpha = [0.1, 1, 10, 100, 1000, 10000]

resultados = []

for alpha in valores_alpha:
    modelo = Ridge(alpha=alpha)
    modelo.fit(X_train_scaled_df, y_train)

    resultados.append({
        'Alpha': alpha,
        'Intercepto': modelo.intercept_,
        'Coeficientes': modelo.coef_
    })

tabela_coeficientes = pd.DataFrame(resultados)
tabela_coeficientes = tabela_coeficientes.set_index('Alpha')


if __name__ == "__main__":
    print("--- COEFICIENTES RIDGE ---")
    print(tabela_coeficientes)

    print("\n--- INTERCEPTOS ---")
    for alpha, intercepto in zip(valores_alpha, tabela_coeficientes['Intercepto']):
        print(f"Alpha {alpha}: {intercepto:.6f}")