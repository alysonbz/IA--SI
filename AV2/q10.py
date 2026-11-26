# ==========================================
# QUESTÃO 10 - Avaliação e Comparação dos Modelos
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from _q8_treinamento import X_test_scaled_df, y_test
from _treino_regressao import reg
from _grid_search_ import melhor_ridge, melhor_lasso

# ---------------------------------------------------------
# 10.a) e 10.b) Cálculo das Métricas e Tabela Comparativa
# ---------------------------------------------------------
# Fazendo as previsões no conjunto de teste
pred_linear = reg.predict(X_test_scaled_df)
pred_ridge = melhor_ridge.predict(X_test_scaled_df)
pred_lasso = melhor_lasso.predict(X_test_scaled_df)

# Função para calcular as 4 métricas
def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    return mae, mse, rmse, r2

# Coletando os resultados
resultados = {
    'Regressão Linear': calcular_metricas(y_test, pred_linear),
    'Ridge': calcular_metricas(y_test, pred_ridge),
    'Lasso': calcular_metricas(y_test, pred_lasso)
}

# Montando a tabela igual ao modelo pedido
tabela_metricas = pd.DataFrame(resultados, index=['MAE', 'MSE', 'RMSE', 'R²']).T
print("\n--- TABELA COMPARATIVA DE MÉTRICAS (Q10.b) ---")
print(tabela_metricas.round(4))

# ---------------------------------------------------------
# 10.c) Gráficos de Valores Reais vs Previstos
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Valores Reais vs Previstos (Y1)', fontsize=16, fontweight='bold')

modelos = [
    ('Regressão Linear', pred_linear, axes[0]),
    ('Ridge', pred_ridge, axes[1]),
    ('Lasso', pred_lasso, axes[2])
]

for nome, pred, ax in modelos:
    ax.scatter(y_test, pred, alpha=0.5, color='royalblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(nome)
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Previsto')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10.d) Análise da Distribuição dos Resíduos
# ---------------------------------------------------------
# ---------------------------------------------------------
# 10.d) Análise da Distribuição dos Resíduos (CORRIGIDO)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 10.d) Análise da Distribuição dos Resíduos (Versão à Prova de Falhas)
# ---------------------------------------------------------
def plotar_residuo_individual(nome, pred):
    residuos = y_test - pred
    plt.figure(figsize=(8, 5)) # Cria uma janela nova para cada modelo
    plt.hist(residuos, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.title(f'Resíduos - {nome}', fontsize=14, fontweight='bold')
    plt.xlabel('Erro (Real - Previsto)')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # Mostra e trava a janela até você fechar

# Gerando um por um (o código vai pausar até você fechar cada janela)
plotar_residuo_individual('Regressão Linear', pred_linear)
plotar_residuo_individual('Ridge', pred_ridge)
plotar_residuo_individual('Lasso', pred_lasso)