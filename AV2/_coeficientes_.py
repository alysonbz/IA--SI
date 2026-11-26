import pandas as pd
from _treino_regressao import reg
from _grid_search_ import melhor_ridge, melhor_lasso
from _preparacao_ import variaveis_preditoras

comparacao = pd.DataFrame({
    'Variável': variaveis_preditoras,
    'Regressão Linear': reg.coef_,
    'Ridge': melhor_ridge.coef_,
    'Lasso': melhor_lasso.coef_
})

print("--- COMPARAÇÃO DOS COEFICIENTES ---")
print(comparacao)

print("\n--- COEFICIENTES ZERADOS ---")

for modelo in ['Regressão Linear', 'Ridge', 'Lasso']:
    quantidade = (comparacao[modelo] == 0).sum()
    print(f"{modelo}: {quantidade}")