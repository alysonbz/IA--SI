# Momento II — Aprendizado supervisionado: regressão
# Resistência à compressão do concreto: previsão do resultado de ensaios
#
# Com que precisão podemos estimar a resistência (MPa) a partir das mesmas oito entradas? Comparamos regressão linear, Ridge e Lasso, e ao final relacionamos os resultados com os clusters do Momento I.



from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # gera as figuras sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

SEMENTE = 42
RAIZ = Path(__file__).resolve().parent  # pasta deste arquivo
DADOS = RAIZ / "dados" / "concreto_uci.csv"
TABELAS = RAIZ / "resultados" / "tabelas"
FIGURAS = RAIZ / "resultados" / "figuras"
for pasta in (DADOS.parent, TABELAS, FIGURAS):
    pasta.mkdir(parents=True, exist_ok=True)

AZUL, LARANJA, TINTA = "#276778", "#bb612d", "#21323c"
CORES_GRUPOS = ["#0072B2", "#008267", "#A65D97", "#C25B12", "#77651D", "#6353A2",
                "#547782", "#955e40", "#547449", "#885668"]
DIVERGENTE = LinearSegmentedColormap.from_list(
    "correlacao", ["#a34b24", "#fcfaf6", "#166775"]
)
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelcolor": TINTA, "text.color": TINTA, "xtick.color": TINTA,
    "ytick.color": TINTA, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#bcc5c8", "axes.axisbelow": True, "figure.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans",
})

def tabela(quadro, nome, casas=2):
    quadro.to_csv(TABELAS / f"{nome}.csv", encoding="utf-8", float_format="%.10g")
    with pd.option_context("display.float_format", lambda x: numero_br(x, casas)):
        print(quadro.fillna("—"))

def figura(fig, nome):
    destino = FIGURAS / f"{nome}.png"
    fig.savefig(destino, dpi=145, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"Figura salva em {destino}")

def numero_br(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")

print(f"pandas {pd.__version__} · NumPy {np.__version__} · "
      f"Matplotlib {mpl.__version__} · scikit-learn {sklearn.__version__}")
# --- Entradas vindas do Momento I (execute o Momento I antes) ---
ingredientes = ["Cimento", "Escória", "Cinza volante", "Água",
                "Superplastificante", "Agregado graúdo", "Agregado miúdo"]
entradas = ingredientes + ["Idade"]
alvo = "Resistência"

base = pd.read_csv(TABELAS / "06_base_antes_da_escala.csv", index_col="ID do ensaio")
atrib = pd.read_csv(TABELAS / "10_atribuicoes.csv", index_col="ID do ensaio")
assert base.index.equals(atrib.index), "Rode o Momento I novamente: os arquivos não coincidem."

X, y = base[entradas], base[alvo]
receita, particao, cluster = atrib["Receita"], atrib["Partição"], atrib["Cluster"]
grupos = sorted(cluster.unique(), key=lambda c: int(c[1:]))

eh_treino = particao.eq("Treino")
X_treino, X_teste = X[eh_treino], X[~eh_treino]
y_treino, y_teste = y[eh_treino], y[~eh_treino]
receitas_treino, receitas_teste = receita[eh_treino], receita[~eh_treino]
assert set(receitas_treino).isdisjoint(receitas_teste)

divisao = pd.DataFrame({
    "Ensaios": [len(X_treino), len(X_teste)],
    "Receitas distintas": [receitas_treino.nunique(), receitas_teste.nunique()],
}, index=pd.Index(["Treino", "Teste"], name="Partição"))
divisao["Percentual dos ensaios"] = 100 * divisao["Ensaios"] / len(base)
print(f"{len(base)} ensaios limpos · {len(grupos)} clusters do Momento I")
tabela(divisao, "07_divisao")


# Divisão treino/teste (item 8a)
# A divisão 80/20 foi definida no Momento I, por receita, e reaproveitada aqui. A tabela acima confirma 804 ensaios de treino e 201 de teste, sem receita em comum.


# 7. Problema de regressão
#
# A variável dependente é a resistência à compressão (MPa), a medição contínua que se deseja antecipar para uma formulação e uma idade de ensaio conhecidas. As variáveis independentes são cimento, escória, cinza volante, água, superplastificante, agregado graúdo, agregado miúdo e idade. O ID do ensaio, o identificador da receita e o cluster servem apenas para organizar os resultados.


corr_treino = pd.concat([X_treino, y_treino], axis=1).corr()
relacao_alvo = corr_treino.loc[entradas, alvo].rename("r com resistência").to_frame()
tabela(relacao_alvo, "13_correlacao_com_alvo_treino", 3)

matriz_x = X_treino.corr()
triangulo = np.triu(np.ones(matriz_x.shape, dtype=bool), k=1)
pares_treino = matriz_x.where(triangulo).stack().rename("r")
pares_treino = pares_treino.reindex(pares_treino.abs().sort_values(ascending=False).index)
pares_treino.index.names = ["Variável 1", "Variável 2"]
pares_treino.to_csv(TABELAS / "14_correlacoes_entradas_treino.csv", float_format="%.8g")
print(pares_treino.head(5).round(3).to_frame())


# No treino, as maiores correlações positivas com o alvo são as de cimento (+0,500), superplastificante (+0,372) e idade (+0,333). A água aparece com −0,324. Entre os preditores, o maior valor absoluto é o de água com superplastificante (−0,657). Nenhum par atinge |r| ≥ 0,80, referência descritiva adotada aqui. Ainda assim, essa associação pode fazer os coeficientes compartilharem informação e variarem quando se aplica regularização. As correlações entre pares também podem ocultar dependências que envolvem vários ingredientes.
#
# A seleção das oito entradas se baseia nas variáveis conhecidas antes da leitura do ensaio. A separação por receita reduz o risco de avaliar uma composição já vista. A análise de correlações acima usa somente o treino.


# 8. Preparação para a regressão
#
# Usamos os mesmos 804 ensaios de treino e 201 de teste para os três modelos. Dentro do treino, a validação cruzada usa cinco dobras de GroupKFold, a versão de K-Fold que mantém cada receita inteira em uma dobra. Em cada ajuste, uma pipeline estima a média e o desvio-padrão só na parte de treinamento da dobra. No reajuste final, o escalonador aprende com os 804 ensaios. O alvo permanece em MPa.


kfold = GroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
dobras = list(kfold.split(X_treino, groups=receitas_treino))
resumo_dobras = []
for i, (ajuste, validacao) in enumerate(dobras, start=1):
    r_ajuste = receitas_treino.iloc[ajuste]
    r_validacao = receitas_treino.iloc[validacao]
    assert set(r_ajuste).isdisjoint(r_validacao)
    resumo_dobras.append([i, len(ajuste), len(validacao), r_validacao.nunique()])
quadro_dobras = pd.DataFrame(resumo_dobras, columns=[
    "Dobra", "Ensaios de ajuste", "Ensaios de validação", "Receitas de validação",
]).set_index("Dobra")
tabela(quadro_dobras, "15_dobras")

def pipeline(modelo):
    return Pipeline([("escala", StandardScaler()), ("modelo", modelo)])


# 9. Treinamento, hiperparâmetros e coeficientes
#
# A regressão linear usa mínimos quadrados ordinários. Ridge acrescenta uma penalidade L2 sobre os coeficientes. Lasso aplica a penalidade L1 e pode zerar alguns deles. Mantemos o intercepto livre nos três modelos.
#
# Para Ridge e Lasso, a busca em grade avalia oito valores de α, de 0,0001 a 1.000. A escolha do valor de α e do modelo segue o menor RMSE médio nas cinco dobras de treino. A tabela mostra o desvio-padrão do RMSE entre dobras, com peso igual para cada dobra.


alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
pontuacao = "neg_root_mean_squared_error"
ajustados = {}
escolhas_cv = []
grades = []

linear = pipeline(LinearRegression(fit_intercept=True))
rmse_linear = -cross_val_score(linear, X_treino, y_treino, cv=dobras, scoring=pontuacao)
ajustados["Linear"] = linear.fit(X_treino, y_treino)
escolhas_cv.append(["Linear", None, rmse_linear.mean(), rmse_linear.std()])

regularizados = {
    "Ridge": Ridge(fit_intercept=True, solver="svd"),
    "Lasso": Lasso(fit_intercept=True, selection="cyclic", max_iter=50000, tol=1e-7),
}
for nome, modelo in regularizados.items():
    busca = GridSearchCV(pipeline(modelo), {"modelo__alpha": alphas}, cv=dobras,
                         scoring=pontuacao, refit=True, n_jobs=1, error_score="raise")
    busca.fit(X_treino, y_treino)
    ajustados[nome] = busca.best_estimator_
    res = busca.cv_results_
    grades.append(pd.DataFrame({
        "Modelo": nome, "Alpha": np.asarray(res["param_modelo__alpha"], dtype=float),
        "RMSE médio": -res["mean_test_score"], "DP entre dobras": res["std_test_score"],
    }))
    escolhas_cv.append([nome, busca.best_params_["modelo__alpha"], -busca.best_score_,
                        res["std_test_score"][busca.best_index_]])

grade = pd.concat(grades, ignore_index=True)
grade.to_csv(TABELAS / "16_grade_completa.csv", index=False, float_format="%.10g")
print(grade.pivot(index="Alpha", columns="Modelo", values="RMSE médio").round(4))
cv = pd.DataFrame(escolhas_cv, columns=["Modelo", "Alpha", "RMSE médio (MPa)",
                                        "DP entre dobras (MPa)"]).set_index("Modelo")
modelo_escolhido = cv["RMSE médio (MPa)"].idxmin()
tabela(cv, "17_selecao_por_validacao", 4)
print(f"Modelo escolhido antes da avaliação no teste: {modelo_escolhido}.")


coeficientes = pd.DataFrame({
    nome: modelo.named_steps["modelo"].coef_ for nome, modelo in ajustados.items()
}, index=pd.Index(entradas, name="Entrada padronizada"))
tabela(coeficientes, "18_coeficientes", 3)
interceptos = {n: round(float(p.named_steps["modelo"].intercept_), 4) for n, p in ajustados.items()}
print("Interceptos em MPa:", interceptos)


# Efeito sobre os pesos. Cada coeficiente indica a variação estimada em MPa para o aumento de um desvio-padrão do respectivo preditor, mantendo as outras entradas fixas. O peso do cimento passa de +11,58 na linear para +11,39 no Ridge (α = 1) e +10,54 no Lasso (α = 0,1). O Ridge mantém as oito entradas com peso diferente de zero. O Lasso zera o peso do agregado graúdo e muda o do agregado miúdo de +0,59 para −0,11. Isso ilustra como pesos pequenos são sensíveis à penalização em uma mistura com ingredientes associados.
#
# O Lasso obteve RMSE de validação de 10,279 MPa, seguido por Ridge (10,295) e linear (10,297). As diferenças são pequenas em relação à variação entre dobras. Escolhemos o Lasso pelo critério definido no treino e por usar sete coeficientes diferentes de zero.


# 10. Avaliação no conjunto de teste
#
# O MAE mede o erro absoluto médio em MPa. O MSE calcula a média dos erros ao quadrado (MPa²). O RMSE devolve essa medida à escala de MPa, com maior peso para erros grandes. O R² mede a fração da variação observada no teste explicada pelas previsões. Valores menores favorecem as três métricas de erro e valores maiores favorecem R².


previsoes = pd.DataFrame({nome: p.predict(X_teste) for nome, p in ajustados.items()},
                         index=X_teste.index)
resultados = []
for nome, previsto in previsoes.items():
    resultados.append([nome, mean_absolute_error(y_teste, previsto),
                       mean_squared_error(y_teste, previsto),
                       root_mean_squared_error(y_teste, previsto), r2_score(y_teste, previsto)])
metricas = pd.DataFrame(resultados, columns=["Modelo", "MAE (MPa)", "MSE (MPa²)",
                                            "RMSE (MPa)", "R²"]).set_index("Modelo")
tabela(metricas, "19_metricas_teste", 4)

cores_modelos = {"Linear": "#58728e", "Ridge": "#977039", "Lasso": "#16786f"}
fig, eixos = plt.subplots(1, 3, figsize=(12.1, 4.2), sharex=True, sharey=True,
                          constrained_layout=True)
lim_inf = np.floor(min(y_teste.min(), previsoes.min().min()) / 5) * 5
lim_sup = np.ceil(max(y_teste.max(), previsoes.max().max()) / 5) * 5
for ax, (nome, previsto) in zip(eixos, previsoes.items()):
    ax.plot([lim_inf, lim_sup], [lim_inf, lim_sup], "--", color="#707d82", lw=1.2)
    ax.scatter(y_teste, previsto, s=19, alpha=.62, color=cores_modelos[nome], edgecolors="none")
    ax.set(xlim=(lim_inf, lim_sup), ylim=(lim_inf, lim_sup), aspect="equal",
           xlabel="Resistência medida (MPa)")
    ax.set_title(f"{nome}   ·   R² = {numero_br(metricas.loc[nome, 'R²'], 3)}", loc="left")
    ax.grid(color="#ebefef")
eixos[0].set_ylabel("Resistência prevista (MPa)")
figura(fig, "07_reais_e_previstos")


# Resultado no teste. A regressão linear ficou em primeiro lugar nas quatro métricas, com MAE de 8,64 MPa, RMSE de 10,98 MPa e R² de 0,546. O Lasso chegou a 11,03 MPa de RMSE, diferença inferior a 0,05 MPa. A regularização mudou os pesos, mas trouxe pouca diferença preditiva nesta divisão. Os pontos abaixo da diagonal representam subestimação. Ela é visível, sobretudo, entre ensaios de resistência elevada.


# Convenção: resíduo positivo significa que o modelo subestimou a resistência.
residuos = previsoes.rsub(y_teste, axis=0)
resumo_res = pd.DataFrame({
    "Média": residuos.mean(), "Mediana": residuos.median(), "DP": residuos.std(),
    "P5": residuos.quantile(.05), "P95": residuos.quantile(.95),
    "Mínimo": residuos.min(), "Máximo": residuos.max(),
}).rename_axis("Modelo")
tabela(resumo_res, "20_residuos_teste")
bins = np.linspace(residuos.min().min() - 1, residuos.max().max() + 1, 17)
fig, eixos = plt.subplots(2, 3, figsize=(12.2, 6.2), sharey="row", constrained_layout=True)
for i, nome in enumerate(previsoes):
    ax, bx = eixos[:, i]
    cor = cores_modelos[nome]
    ax.hist(residuos[nome], bins=bins, color=cor, edgecolor="white")
    ax.axvline(0, color=TINTA, ls="--", lw=1.1)
    ax.set_title(nome, loc="left")
    ax.set_xlabel("Resíduo (MPa)")
    bx.scatter(previsoes[nome], residuos[nome], s=16, color=cor, alpha=.55, edgecolors="none")
    bx.axhline(0, color=TINTA, ls="--", lw=1.1)
    bx.set(xlabel="Resistência prevista (MPa)", xlim=(lim_inf, lim_sup))
    ax.grid(axis="y", color="#edf0f1")
    bx.grid(axis="y", color="#edf0f1")
eixos[0, 0].set_ylabel("Ensaios")
eixos[1, 0].set_ylabel("Medido − previsto (MPa)")
figura(fig, "08_diagnostico_residuos")

por_ensaio = pd.concat([receitas_teste, cluster.loc[y_teste.index], X_teste["Idade"],
                        y_teste.rename("Medido (MPa)"), previsoes.add_prefix("Previsto · "),
                        residuos.add_prefix("Resíduo · ")], axis=1)
por_ensaio.to_csv(TABELAS / "21_previsoes_por_ensaio.csv", float_format="%.10g")


# Os resíduos médios ficaram entre +0,15 e +0,18 MPa, quase centrados em zero, embora a dispersão seja de aproximadamente 11 MPa. Para o Lasso, os percentis 5 e 95 são −18,35 e +18,79 MPa. Logo, o pequeno viés médio convive com erros individuais consideráveis, em ambos os sentidos.
#
# Quando usar a regularização? Ridge pode estabilizar estimativas quando vários preditores relacionados contêm informação útil. Lasso pode simplificar um modelo ao zerar contribuições redundantes. Esta base exemplifica a segunda situação com o agregado graúdo. Como os três modelos usam uma função linear e aditiva das oito entradas, nenhuma das penalidades representa diretamente uma curva de maturação ou uma interação entre dosagens.


# 11. Análise integrada
#
# Agora associamos o alvo medido aos grupos definidos só com composição e idade. A distribuição da resistência descreve os 1.005 ensaios limpos. Para avaliar o erro em cada perfil, usamos apenas as previsões dos 201 ensaios reservados e o Lasso escolhido pela validação.


alvo_grupos = y.groupby(cluster).describe(percentiles=[.25, .50, .75]).reindex(grupos)
alvo_grupos.columns = ["Ensaios", "Média", "DP", "Mínimo", "Q1", "Mediana", "Q3", "Máximo"]
alvo_grupos["Ensaios"] = alvo_grupos["Ensaios"].astype(int)
tabela(alvo_grupos, "22_resistencia_por_cluster")

fig, ax = plt.subplots(figsize=(9.2, 4.1), constrained_layout=True)
caixas = ax.boxplot([y.loc[cluster.eq(g)] for g in grupos], tick_labels=grupos,
                    patch_artist=True, widths=.55, medianprops={"color": TINTA, "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 3, "markeredgecolor": "#758489"})
for caixa, cor in zip(caixas["boxes"], CORES_GRUPOS):
    caixa.set(facecolor=cor, alpha=.50)
ax.set(xlabel="Cluster", ylabel="Resistência medida (MPa)")
ax.grid(axis="y", color="#e5eaeb")
figura(fig, "09_resistencia_nos_clusters")


erros_grupo = []
for grupo in grupos:
    ids = X_teste.index[cluster.loc[X_teste.index].eq(grupo)]
    for nome in previsoes:
        medido, previsto = y_teste.loc[ids], previsoes.loc[ids, nome]
        erros_grupo.append([grupo, nome, len(ids), mean_absolute_error(medido, previsto),
                            root_mean_squared_error(medido, previsto), (medido - previsto).mean()])
por_grupo = pd.DataFrame(erros_grupo, columns=[
    "Cluster", "Modelo", "Ensaios no teste", "MAE (MPa)", "RMSE (MPa)", "Viés médio (MPa)",
])
por_grupo.to_csv(TABELAS / "23_erros_por_cluster_todos_modelos.csv", index=False, float_format="%.10g")
erros_final = por_grupo.loc[por_grupo["Modelo"].eq(modelo_escolhido)].drop(columns="Modelo").set_index("Cluster")
tabela(erros_final, "24_erros_por_cluster_modelo_escolhido")

fig, ax = plt.subplots(figsize=(9.1, 3.9), constrained_layout=True)
pos = np.arange(len(grupos))
for coluna, dx, cor in [("MAE (MPa)", -.19, AZUL), ("RMSE (MPa)", .19, LARANJA)]:
    barras = ax.bar(pos + dx, erros_final[coluna], width=.36, color=cor, label=coluna.split()[0])
    ax.bar_label(barras, labels=[numero_br(v, 1) for v in erros_final[coluna]], fontsize=9, padding=3)
ax.set(xticks=pos, xticklabels=[f"{g}\n(n={n})" for g, n in zip(grupos, erros_final["Ensaios no teste"])],
       ylabel="Erro no teste (MPa)", ylim=(0, 17.5))
ax.legend(loc="upper left", ncols=2, frameon=False)
ax.grid(axis="y", color="#e5eaeb")
figura(fig, "10_erro_por_cluster")

# A média ponderada dos MAEs de grupo recupera o MAE global; para RMSE, agregam-se os quadrados.
pesos = erros_final["Ensaios no teste"]
assert pesos.sum() == len(y_teste)
assert np.isclose(np.average(erros_final["MAE (MPa)"], weights=pesos),
                  metricas.loc[modelo_escolhido, "MAE (MPa)"])
assert np.isclose(np.sqrt(np.average(erros_final["RMSE (MPa)"] ** 2, weights=pesos)),
                  metricas.loc[modelo_escolhido, "RMSE (MPa)"])


# a) Quais padrões apareceram? C1 se destaca pela escória. C2 e C3 compartilham o uso de cinza, com dosagens diferentes de agregado graúdo e água. C4 reúne ensaios em idades avançadas. C5 e C6 se diferenciam principalmente pelas adições, pelo teor de cimento e pela massa de água por metro cúbico de mistura.
#
# b) As características dos clusters também se relacionam com a resistência? Sim. C6 reúne mais cimento e aditivo, além de menos água. No treino, as duas primeiras variáveis se associam positivamente à resistência e a água se associa negativamente. A idade diferencia C4 e também tem correlação positiva com o alvo. As associações descrevem combinações de ingredientes e condições observadas e permitem formular hipóteses.
#
# c) A resistência difere entre os grupos? C6 tem a maior mediana (54,90 MPa). C5 tem a menor (27,19 MPa), embora a mediana da idade seja 28 dias nos dois grupos. As faixas entre o primeiro e o terceiro quartil são de 41,37 a 64,30 MPa em C6 e de 17,57 a 39,05 MPa em C5. Ainda há sobreposição nos intervalos completos e ampla variação dentro de cada grupo.
#
# d) O erro é semelhante em todos os grupos? C5 tem os menores MAE e RMSE do Lasso (6,77 e 8,79 MPa). C4 tem o maior MAE (11,60 MPa) e C6 o maior RMSE (13,99 MPa). O viés de −5,59 MPa em C4 indica superestimação em média. Como C4 contém apenas 12 ensaios de teste e C6 contém 19, poucos erros grandes podem alterar a ordenação.
#
# e) O que se conclui? A composição e a idade revelam grupos interpretáveis e explicam parte da variação da resistência. O Lasso explica 54,2% da variação dos resultados no teste e erra em média 8,70 MPa em valor absoluto. A regressão linear teve uma pequena vantagem no teste, e o Lasso usou uma entrada a menos com desempenho próximo. A dispersão dentro dos clusters e os erros maiores em alguns perfis delimitam o alcance dessas previsões para receitas inéditas. Estas conclusões dependem desta amostra e de uma divisão de teste.