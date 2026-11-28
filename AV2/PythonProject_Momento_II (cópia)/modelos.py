
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. IMPORTACAO DOS DADOS E DO PRE-PROCESSADOR
# ==========================================

from pre_processamento import (
    X_train,
    X_test,
    y_train,
    y_test,
    criar_preprocessador
)

print("\n==========================================")
print("SECAO 9 - MODELOS DE REGRESSAO")
print("==========================================")

# Todos os modelos utilizarao a mesma divisao
# de treinamento e teste da Secao 8.


# ==========================================
# 2. CONFIGURACAO DA VALIDACAO CRUZADA
# ==========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Valores de alpha que serao testados

valores_alpha = [0.01, 0.1, 1, 10, 100]

print("\nValores de alpha testados:")
print(valores_alpha)


# ==========================================
# 3. MODELO DE REGRESSAO LINEAR
# ==========================================

print("\n==========================================")
print("MODELO 1 - REGRESSAO LINEAR")
print("==========================================")

modelo_linear = Pipeline(
    steps=[
        ("preprocessamento", criar_preprocessador()),
        ("modelo", LinearRegression())
    ]
)

modelo_linear.fit(X_train, y_train)

previsoes_linear = modelo_linear.predict(X_test)

print("\nRegressao Linear treinada com sucesso!")


# ==========================================
# 4. MODELO RIDGE COM GRIDSEARCHCV
# ==========================================

print("\n==========================================")
print("MODELO 2 - RIDGE")
print("==========================================")

modelo_ridge = Pipeline(
    steps=[
        ("preprocessamento", criar_preprocessador()),
        ("modelo", Ridge())
    ]
)

parametros_ridge = {
    "modelo__alpha": valores_alpha
}

busca_ridge = GridSearchCV(
    estimator=modelo_ridge,
    param_grid=parametros_ridge,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

busca_ridge.fit(X_train, y_train)

melhor_ridge = busca_ridge.best_estimator_

previsoes_ridge = melhor_ridge.predict(X_test)

print("\nMelhor configuracao Ridge:")
print(busca_ridge.best_params_)

print("\nMelhor resultado medio na validacao cruzada (RMSE):")

melhor_rmse_ridge = np.sqrt(-busca_ridge.best_score_)

print(round(melhor_rmse_ridge, 3))


# ==========================================
# 5. MODELO LASSO COM GRIDSEARCHCV
# ==========================================

print("\n==========================================")
print("MODELO 3 - LASSO")
print("==========================================")

modelo_lasso = Pipeline(
    steps=[
        ("preprocessamento", criar_preprocessador()),
        ("modelo", Lasso(max_iter=20000))
    ]
)

parametros_lasso = {
    "modelo__alpha": valores_alpha
}

busca_lasso = GridSearchCV(
    estimator=modelo_lasso,
    param_grid=parametros_lasso,
    cv=kf,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

busca_lasso.fit(X_train, y_train)

melhor_lasso = busca_lasso.best_estimator_

previsoes_lasso = melhor_lasso.predict(X_test)

print("\nMelhor configuracao Lasso:")
print(busca_lasso.best_params_)

print("\nMelhor resultado medio na validacao cruzada (RMSE):")

melhor_rmse_lasso = np.sqrt(-busca_lasso.best_score_)

print(round(melhor_rmse_lasso, 3))


# ==========================================
# 6. AVALIACAO DOS MODELOS
# ==========================================

print("\n==========================================")
print("COMPARACAO DOS MODELOS")
print("==========================================")

def avaliar_modelo(nome, y_real, y_previsto):

    mae = mean_absolute_error(y_real, y_previsto)

    rmse = np.sqrt(
        mean_squared_error(y_real, y_previsto)
    )

    r2 = r2_score(y_real, y_previsto)

    return {
        "Modelo": nome,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


resultados = []

resultados.append(
    avaliar_modelo(
        "Regressao Linear",
        y_test,
        previsoes_linear
    )
)

resultados.append(
    avaliar_modelo(
        "Ridge",
        y_test,
        previsoes_ridge
    )
)

resultados.append(
    avaliar_modelo(
        "Lasso",
        y_test,
        previsoes_lasso
    )
)

tabela_resultados = pd.DataFrame(resultados)

print("\nMetricas no conjunto de teste:")
print(tabela_resultados.round(3).to_string(index=False))


# ==========================================
# 7. COMPARACAO DOS COEFICIENTES
# ==========================================

print("\n==========================================")
print("COMPARACAO DOS COEFICIENTES")
print("==========================================")

# Recuperar os nomes das caracteristicas
# depois da codificacao OneHotEncoder.

preprocessador_linear = (
    modelo_linear.named_steps["preprocessamento"]
)

nomes_caracteristicas = (
    preprocessador_linear
    .get_feature_names_out()
)

coeficientes = pd.DataFrame({
    "Caracteristica": nomes_caracteristicas,
    "Linear": modelo_linear.named_steps["modelo"].coef_,
    "Ridge": melhor_ridge.named_steps["modelo"].coef_,
    "Lasso": melhor_lasso.named_steps["modelo"].coef_
})

print("\nPrimeiros coeficientes:")
print(coeficientes.head(15).round(4).to_string(index=False))


# ==========================================
# 8. IDENTIFICACAO DOS COEFICIENTES ZERADOS
# ==========================================

print("\n==========================================")
print("COEFICIENTES ZERADOS PELO LASSO")
print("==========================================")

coef_lasso = melhor_lasso.named_steps["modelo"].coef_

quantidade_zerados = np.sum(
    np.isclose(coef_lasso, 0, atol=1e-8)
)

quantidade_total = len(coef_lasso)

print("\nQuantidade de coeficientes zerados:", quantidade_zerados)

print("Quantidade total de coeficientes:", quantidade_total)

print(
    "Percentual de coeficientes zerados:",
    round(100 * quantidade_zerados / quantidade_total, 2),
    "%"
)

print("\nCaracteristicas com coeficiente zerado:")

for nome, coef in zip(nomes_caracteristicas, coef_lasso):

    if np.isclose(coef, 0, atol=1e-8):
        print("-", nome)


# ==========================================
# 9. SALVAR OS RESULTADOS
# ==========================================

tabela_resultados.to_csv(
    "resultados/comparacao_modelos.csv",
    index=False
)

coeficientes.to_csv(
    "resultados/comparacao_coeficientes.csv",
    index=False
)

print("\n==========================================")
print("RESUMO FINAL")
print("==========================================")

print("\nMelhor alpha Ridge:")
print(busca_ridge.best_params_)

print("\nMelhor alpha Lasso:")
print(busca_lasso.best_params_)

print("\nResultados e coeficientes salvos na pasta resultados.")

print("\nExecucao dos modelos de regressao concluida!")
