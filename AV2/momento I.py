# Momento I — Aprendizado não supervisionado (K-Means)
# Resistência à compressão do concreto: agrupamento de misturas
#
# Quais misturas de concreto têm perfis semelhantes? O K-Means responde a essa pergunta a partir de sete dosagens e da idade do ensaio. A resistência (MPa) não entra no agrupamento: ela fica reservada para o Momento II.


from pathlib import Path
from urllib.request import urlopen

import matplotlib as mpl

mpl.use("Agg")  # gera as figuras sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

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


# 3. Análise exploratória
#
# Cada linha descreve um ensaio. As dosagens representam a massa de cada ingrediente em um metro cúbico da mistura. A idade indica quantos dias o concreto tinha no ensaio. Esta seção descreve a base original, inclusive as repetições, para documentar o que foi recebido.


URL_CSV = "https://archive.ics.uci.edu/static/public/165/data.csv"
if not DADOS.exists():
    with urlopen(URL_CSV, timeout=30) as resposta:
        DADOS.write_bytes(resposta.read())

nomes = {
    "Cement": "Cimento", "Blast Furnace Slag": "Escória",
    "Fly Ash": "Cinza volante", "Water": "Água",
    "Superplasticizer": "Superplastificante", "Coarse Aggregate": "Agregado graúdo",
    "Fine Aggregate": "Agregado miúdo", "Age": "Idade",
    "Concrete compressive strength": "Resistência",
}
original = pd.read_csv(DADOS)
assert list(original.columns) == list(nomes), "Confira as colunas do CSV oficial."
original = original.rename(columns=nomes).apply(pd.to_numeric, errors="raise")
original.index = pd.RangeIndex(1, len(original) + 1, name="ID do ensaio")
ingredientes = list(nomes.values())[:7]
entradas = ingredientes + ["Idade"]
alvo = "Resistência"
unidades = {**dict.fromkeys(ingredientes, "kg/m³"), "Idade": "dias", alvo: "MPa"}

dicionario = pd.DataFrame({
    "Papel": ["Entrada"] * len(entradas) + ["Alvo"],
    "Unidade": pd.Series(unidades), "Tipo no CSV": original.dtypes.astype(str),
    "Valores distintos": original.nunique(), "Ausentes": original.isna().sum(),
})
dicionario.index.name = "Variável"
print(f"{len(original)} ensaios · {len(entradas)} entradas · 1 alvo numérico")
tabela(dicionario, "01_dicionario")


# A idade é uma contagem inteira. As sete dosagens e a resistência são medidas quantitativas contínuas. A escória, um subproduto de alto-forno, contém frações de kg/m³ no CSV, como 142,5. A análise preserva essas casas decimais.


duplicata = original.duplicated(keep="first")
opcionais = ["Escória", "Cinza volante", "Superplastificante"]
obrigatorias = ["Cimento", "Água", "Agregado graúdo", "Agregado miúdo"]
checagens = pd.Series({
    "Células ausentes": int(original.isna().sum().sum()),
    "Células infinitas": int(np.isinf(original.to_numpy()).sum()),
    "Linhas com dosagem negativa": int(original[ingredientes].lt(0).any(axis=1).sum()),
    "Linhas com zero em componente básico": int(original[obrigatorias].eq(0).any(axis=1).sum()),
    "Linhas com idade fracionária ou fora de 1–365 dias":
        int((~original["Idade"].between(1, 365) | original["Idade"].mod(1).ne(0)).sum()),
    "Linhas com resistência não positiva": int(original[alvo].le(0).sum()),
    "Linhas em algum conjunto de cópias idênticas":
        int(original.duplicated(keep=False).sum()),
    "Cópias exatas além da primeira": int(duplicata.sum()),
}, name="Quantidade").rename_axis("Verificação").to_frame()
tabela(checagens, "02_qualidade")
print("Ensaios com dosagem zero:", original[opcionais].eq(0).sum().to_dict())

resumo = original.agg(["mean", "median", "std", "min", "max"]).T
resumo.columns = ["Média", "Mediana", "DP amostral", "Mínimo", "Máximo"]
resumo.insert(0, "Unidade", pd.Series(unidades))
resumo.index.name = "Variável"
tabela(resumo, "03_descritivas_originais", 3)


fig, eixos = plt.subplots(3, 3, figsize=(11.2, 7.9), constrained_layout=True)
for ax, nome in zip(eixos.flat, original):
    cor = LARANJA if nome == alvo else AZUL
    ax.hist(original[nome], bins=22, color=cor, edgecolor="white", linewidth=0.6)
    ax.set_title(f"{nome} ({unidades[nome]})", loc="left", fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.grid(axis="y", color="#e7ebec")
for ax in eixos[:, 0]:
    ax.set_ylabel("Ensaios")
figura(fig, "01_distribuicoes")


# Leitura das distribuições. Há 425 ensaios aos 28 dias, cerca de 41% da base. Poucos ensaios chegam a 180 dias ou mais, o que desloca a média da idade para 45,66 dias, acima da mediana de 28 dias. A dosagem de cinza volante é zero em 566 dos 1.030 ensaios, o que explica sua mediana zero. Os zeros descrevem uma escolha de composição. A resistência varia de 2,33 a 82,60 MPa, com mediana de 34,445 MPa.


curtos = ["Cimento", "Escória", "Cinza", "Água", "Superplast.", "Ag. graúdo",
          "Ag. miúdo", "Idade", "Resistência"]
corr = original.corr(method="pearson")
corr.to_csv(TABELAS / "04_correlacoes_originais.csv", float_format="%.8g")
fig, ax = plt.subplots(figsize=(8.9, 6.0), constrained_layout=True)
mapa = ax.imshow(corr, vmin=-1, vmax=1, cmap=DIVERGENTE, aspect="auto")
ax.set_xticks(range(len(curtos)), curtos, rotation=32, ha="right")
ax.set_yticks(range(len(curtos)), curtos)
for i, linha in enumerate(corr.to_numpy()):
    for j, r in enumerate(linha):
        ax.text(j, i, numero_br(r), ha="center", va="center", fontsize=9,
                color="white" if abs(r) > .61 else TINTA)
fig.colorbar(mapa, ax=ax, shrink=.85, label="Correlação de Pearson (r)")
figura(fig, "02_correlacoes")

fig, eixos = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
pares = [("Cimento", alvo), ("Idade", alvo), ("Superplastificante", "Água")]
for ax, (x, y) in zip(eixos, pares):
    ax.scatter(original[x], original[y], s=13, color=AZUL, alpha=.29,
               edgecolors="none")
    ax.set(xlabel=f"{x} ({unidades[x]})", ylabel=f"{y} ({unidades[y]})")
    ax.grid(color="#edf0f1")
medianas = original.groupby("Idade")[alvo].median()
eixos[1].plot(medianas.index, medianas.values, color=LARANJA, marker="o",
              markersize=4, linewidth=1.5, label="Mediana por idade")
eixos[1].set_xscale("log")
eixos[1].set_xticks([1, 3, 7, 14, 28, 56, 180, 365], ["1", "3", "7", "14", "28", "56", "180", "365"])
eixos[1].set_xlabel("Idade (dias, escala log)")
eixos[1].legend(frameon=False, fontsize=9)
figura(fig, "03_dispersoes")


# Leitura das relações. Na base original, cimento e resistência têm correlação de +0,50. Ainda assim, uma mesma dosagem de cimento aparece em ensaios com resistências bastante diferentes. A mediana aos 7 dias é 21,65 MPa e a dos 28 dias é 33,76 MPa. As misturas ensaiadas variam de uma idade para outra, o que limita a comparação direta. Água e superplastificante têm correlação de −0,66. As misturas com mais aditivo tendem a usar menos água.


q1, q3 = original.quantile(.25), original.quantile(.75)
aiq = q3 - q1
inferior, superior = q1 - 1.5 * aiq, q3 + 1.5 * aiq
fora = original.lt(inferior) | original.gt(superior)
extremos = pd.DataFrame({"Limite inferior": inferior, "Limite superior": superior,
                         "Ensaios sinalizados": fora.sum()})
extremos.index.name = "Variável"
tabela(extremos, "05_extremos_tukey")
print(f"{fora.any(axis=1).sum()} ensaios fora dos limites de Tukey em pelo menos uma variável.")


# Valores discrepantes. A regra de Tukey usa 1,5 vez a amplitude interquartil, medida em cada variável. Há 59 idades acima de 129,5 dias e quatro resistências acima do limite superior. Os limites são estatísticos. As idades registradas estão dentro do intervalo documentado pela UCI. Esses ensaios permanecem na análise. Pontos afastados podem deslocar os centroides do K-Means e exercer maior influência sobre o ajuste por mínimos quadrados. A comparação de MAE e RMSE permite examinar o peso dos grandes erros no teste.


# 4. Pré-processamento e separação das amostras
#
# Mantivemos as sete dosagens e a idade. Juntas, elas descrevem tanto o traço quanto o momento do ensaio, duas fontes de variação relevantes para o problema. A resistência fica reservada como alvo da regressão e para a comparação posterior entre grupos. Todas as entradas são numéricas.
#
# Removemos somente cópias iguais nas nove colunas para evitar peso extra na análise. Sem um identificador de origem, algumas cópias podem corresponder a repetições experimentais legítimas. Ensaios com as mesmas entradas e diferentes resistências são preservados. O código mantém a linha de origem e grava a base limpa nas unidades originais.


base = original.loc[~duplicata].copy()
base.to_csv(TABELAS / "06_base_antes_da_escala.csv", encoding="utf-8")
X = base.loc[:, entradas]
y = base[alvo]
receita = base.groupby(ingredientes, sort=True).ngroup().rename("Receita")
repeticoes_x = base.loc[base.duplicated(entradas, keep=False), entradas]
print(f"{len(original)} linhas originais − {duplicata.sum()} cópias = {len(base)} ensaios.")
print(f"{receita.nunique()} receitas distintas pelas sete dosagens.")
print(f"{len(repeticoes_x.drop_duplicates())} combinações de entradas têm "
      f"resistências diferentes, envolvendo {len(repeticoes_x)} ensaios.")


# Uma receita pode aparecer em mais de uma idade. Para avaliar a previsão para receitas inéditas, todos os ensaios de uma mesma composição ficam no mesmo conjunto. O sorteio aceita a primeira divisão que coloca exatamente 20% das linhas no teste. O critério depende apenas do número de linhas por receita. As duas etapas usam a mesma reserva. As escalas, a escolha de K e a seleção do modelo são ajustadas no treino.


particoes = GroupShuffleSplit(n_splits=2000, test_size=.20, random_state=SEMENTE)
n_teste = round(.20 * len(base))
pos_treino, pos_teste = next(
    (t, s) for t, s in particoes.split(X, groups=receita) if len(s) == n_teste
)
X_treino, X_teste = X.iloc[pos_treino], X.iloc[pos_teste]
y_treino, y_teste = y.loc[X_treino.index], y.loc[X_teste.index]
receitas_treino, receitas_teste = receita.loc[X_treino.index], receita.loc[X_teste.index]
assert set(receitas_treino).isdisjoint(receitas_teste)
particao = pd.Series("Treino", index=base.index, name="Partição")
particao.loc[X_teste.index] = "Teste"
divisao = pd.DataFrame({
    "Ensaios": [len(X_treino), len(X_teste)],
    "Receitas distintas": [receitas_treino.nunique(), receitas_teste.nunique()],
}, index=pd.Index(["Treino", "Teste"], name="Partição"))
divisao["Percentual dos ensaios"] = 100 * divisao["Ensaios"] / len(base)
tabela(divisao, "07_divisao")

escala_cluster = StandardScaler().fit(X_treino)
Z_treino = escala_cluster.transform(X_treino)


# Por que padronizar? O K-Means atribui cada observação ao centroide de menor distância euclidiana e minimiza as distâncias ao quadrado. Na escala original, o desvio-padrão do cimento é 104,51 kg/m³ e o do aditivo é 5,97 kg/m³. Uma variação típica do cimento pesaria muito mais. Usamos $z=(x-\mu_{treino})/\sigma_{treino}$ para que uma mudança de um desvio-padrão tenha a mesma escala numérica em cada entrada. Nas regressões, a padronização também torna comparáveis as penalizações sobre os coeficientes.


# 5. Aplicação do K-Means
#
# Testamos K de 2 a 10 com distância euclidiana, inicialização k-means++ e 50 reinicializações por valor de K. A inércia é a soma dos quadrados das distâncias ao centro do grupo e tende a cair à medida que K aumenta. A silhueta média varia de −1 a +1 e compara a proximidade de cada ponto com seu grupo e com o grupo vizinho. O critério de seleção é a maior silhueta média no treino. O cotovelo ajuda a interpretar o ganho de acrescentar grupos.


def ajustar_kmeans(k):
    return KMeans(n_clusters=k, init="k-means++", n_init=50, algorithm="lloyd",
                  max_iter=300, tol=1e-4, random_state=SEMENTE).fit(Z_treino)

avaliacoes_k = []
# Uma thread mantém previsíveis a ordem das somas e o tempo neste dataset pequeno.
with threadpool_limits(limits=1):
    for k in range(2, 11):
        candidato = ajustar_kmeans(k)
        avaliacoes_k.append({
            "K": k, "Inércia": candidato.inertia_,
            "Silhueta": silhouette_score(Z_treino, candidato.labels_, metric="euclidean"),
        })
criterios = pd.DataFrame(avaliacoes_k).set_index("K")
criterios["Queda de inércia (%)"] = -100 * criterios["Inércia"].pct_change()
k_final = int(criterios["Silhueta"].idxmax())
tabela(criterios, "08_selecao_k", 4)

fig, eixos = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
for ax, coluna, titulo in zip(eixos, ["Inércia", "Silhueta"],
                               ["Método do cotovelo", "Separação entre grupos"]):
    ax.plot(criterios.index, criterios[coluna], color=AZUL, marker="o")
    ax.axvline(k_final, color=LARANJA, linestyle="--", linewidth=1)
    ax.scatter(k_final, criterios.loc[k_final, coluna], color=LARANJA, s=70, zorder=3)
    ax.set(xlabel="Número de clusters (K)", ylabel=coluna, xticks=list(criterios.index))
    ax.set_title(titulo, loc="left")
    ax.grid(axis="y", color="#e5eaeb")
figura(fig, "04_escolha_k")


# Escolha de K. A maior silhueta ocorre em K = 6 (0,2814). A redução da inércia passa de 12,14% na mudança de cinco para seis grupos para 7,11% na mudança de seis para sete. K = 9 chega perto na silhueta (0,2792) e exige descrever três perfis adicionais. Assim, seis grupos conciliam o máximo observado com uma divisão interpretável. O valor absoluto da silhueta indica que várias misturas ficam próximas das fronteiras entre grupos.


with threadpool_limits(limits=1):
    kmeans = ajustar_kmeans(k_final)
    rotulos_originais = kmeans.predict(escala_cluster.transform(X))
# Nomes estáveis para a apresentação, em ordem crescente de cimento no centroide.
ordem = np.argsort(kmeans.cluster_centers_[:, entradas.index("Cimento")])
rotular = {int(rotulo): f"C{i}" for i, rotulo in enumerate(ordem, start=1)}
cluster = pd.Series([rotular[int(r)] for r in rotulos_originais], index=X.index, name="Cluster")
grupos = list(rotular.values())

tamanhos = pd.crosstab(cluster, particao).reindex(index=grupos, columns=["Treino", "Teste"])
tamanhos["Total"] = tamanhos.sum(axis=1)
tamanhos["% da base limpa"] = 100 * tamanhos["Total"] / len(base)
tabela(tamanhos, "09_tamanhos_clusters")
pd.concat([receita, particao, cluster], axis=1).to_csv(TABELAS / "10_atribuicoes.csv")


# 6. Interpretação dos clusters
#
# A tabela mostra médias dos ensaios atribuídos a cada grupo, incluindo treino e teste, nas unidades originais. Os centros usados para classificá-los foram aprendidos nas 804 linhas do treino. C1 a C6 seguem a ordem crescente de cimento nos centros, uma convenção para identificar os grupos.


medias_grupos = X.groupby(cluster).mean().reindex(grupos)
perfis = medias_grupos.T.copy()
perfis.insert(0, "Unidade", pd.Series(unidades))
perfis.index.name = "Variável"
tabela(perfis, "11_medias_dos_clusters")
idade_por_grupo = X["Idade"].groupby(cluster).agg(["min", "median", "max"]).astype(int)
idade_por_grupo.columns = ["Idade mínima (dias)", "Mediana (dias)", "Idade máxima (dias)"]
tabela(idade_por_grupo.reindex(grupos), "11b_idade_nos_clusters")

centros_z = pd.DataFrame(kmeans.cluster_centers_[ordem], index=grupos, columns=entradas)
centros_z.to_csv(TABELAS / "12_centroides_padronizados.csv", index_label="Cluster")
fig, ax = plt.subplots(figsize=(10.3, 4.3), constrained_layout=True)
limite = max(1, np.abs(centros_z.to_numpy()).max())
mapa = ax.imshow(centros_z, cmap=DIVERGENTE, vmin=-limite, vmax=limite, aspect="auto")
ax.set_xticks(range(len(entradas)), curtos[:-1], rotation=24, ha="right")
ax.set_yticks(range(k_final), grupos)
for i, linha in enumerate(centros_z.to_numpy()):
    for j, v in enumerate(linha):
        ax.text(j, i, numero_br(v, 1), ha="center", va="center",
                color="white" if abs(v) > .63 * limite else TINTA)
fig.colorbar(mapa, ax=ax, shrink=.92, label="Desvios-padrão da média do treino")
figura(fig, "05_perfis_clusters")


pca = PCA(n_components=2, svd_solver="full").fit(Z_treino)
projecao = pca.transform(escala_cluster.transform(X))
centros_2d = pca.transform(centros_z.to_numpy())
variancia = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
for i, (grupo, cor) in enumerate(zip(grupos, CORES_GRUPOS)):
    pontos = projecao[cluster.eq(grupo).to_numpy()]
    ax.scatter(pontos[:, 0], pontos[:, 1], s=15, alpha=.50, color=cor,
               edgecolors="none", label=grupo)
    cx, cy = centros_2d[i]
    ax.scatter(cx, cy, marker="X", s=135, color=cor, edgecolors="black", linewidths=1.1)
ax.set(xlabel=f"Componente 1 ({numero_br(variancia[0], 1)}% da variância)",
       ylabel=f"Componente 2 ({numero_br(variancia[1], 1)}% da variância)")
ax.grid(color="#e9edef")
ax.legend(title="Grupo", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
figura(fig, "06_projecao_clusters")


# | Grupo | Perfil observado nas unidades originais |
# | --- | --- |
# | C1 · Maior dosagem média de escória | A média de escória é 188,3 kg/m³, a mais alta. Há pouca cinza volante (1,2 kg/m³) e o menor consumo médio de cimento (211,2 kg/m³). |
# | C2 · Cinza e agregado graúdo | Tem 125,5 kg/m³ de cinza, 1.018,8 kg/m³ de agregado graúdo e 168,1 kg/m³ de água. Usa bem menos escória que C3. |
# | C3 · Cinza combinada com escória | A cinza também é alta (122,4 kg/m³), com mais escória (90,6 kg/m³) e água (192,6 kg/m³) que C2. É o grupo com menos agregado graúdo (877,7 kg/m³). |
# | C4 · Ensaios em idades avançadas | Reúne ensaios de 180 a 365 dias, com mediana de 270 dias. Também concentra bastante água (211,5 kg/m³) e tem superplastificante igual a zero. |
# | C5 · Cimento com poucas adições | Tem 357,7 kg/m³ de cimento e médias próximas de zero para escória, cinza e superplastificante. A mediana de idade é 28 dias. |
# | C6 · Mais cimento e aditivo, menos água | Combina as maiores médias de cimento (406,1 kg/m³) e superplastificante (14,3 kg/m³) com a menor média de água (156,4 kg/m³). |
#
# Na projeção por componentes principais (PCA), cada ponto representa um ensaio e cada X localiza um centroide. As duas dimensões exibem 46,6% da variância do treino. Parte da separação em oito dimensões fica invisível. A distribuição desigual das idades e a maneira de ponderar as oito entradas também influenciam os grupos. Esses nomes resumem os padrões desta amostra.