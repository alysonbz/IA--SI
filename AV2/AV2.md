Z# **AV2 – Inteligência Artificial**

## **1. Objetivo**

O trabalho tem como objetivo aplicar técnicas de aprendizado de máquina supervisionado e não supervisionado sobre um mesmo conjunto de dados, envolvendo análise exploratória, identificação de padrões, agrupamento de amostras e construção de modelos preditivos.

O desenvolvimento será organizado em dois momentos:

- **Momento I – Aprendizado não supervisionado:** análise exploratória dos dados, preparação do dataset, aplicação do algoritmo K-Means, seleção do número de clusters e interpretação dos agrupamentos.
- **Momento II – Aprendizado supervisionado:** seleção de uma variável numérica como alvo de predição, treinamento de modelos de regressão Linear, Ridge e Lasso, avaliação quantitativa e comparação dos resultados.

Os dois momentos deverão utilizar o mesmo dataset, permitindo relacionar os padrões identificados na análise não supervisionada com os resultados obtidos na etapa de regressão.

## **2.  dataset**

#### Duplas

Levi de Sousa Nascimento 

Kemilly da Conceição Mota

Maria Laísa Viana Gomes

dataset: https://archive.ics.uci.edu/dataset/242/energy+efficiency


Francisco Emanuel de Sousa Oliveira

Francisco Kauan Lima Moura

dataset: https://archive.ics.uci.edu/dataset/165/concrete-compressive-strength

Camila Araújo

gabriely

Yasmin Coelho

dataset: https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset

João Pedro 

Thiago Pinto Gomes

dataset: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

Joao victor 

Antonio erick

dataset: https://archive.ics.uci.edu/dataset/320/student-performance

João Gilberto

Marcone Silva

dataset: https://archive.ics.uci.edu/dataset/360/air+quality


Habacuqui

dataset: https://archive.ics.uci.edu/dataset/9/auto+mpg


# **Momento I – Aprendizado não supervisionado**

## **3. Análise exploratória dos dados**

A equipe deverá realizar uma análise exploratória do dataset, buscando compreender sua estrutura, distribuição e características.

### **Atividades obrigatórias**

a) Apresentar o número de amostras e características existentes no dataset, identificando os tipos de dados.

b) Identificar valores ausentes, registros duplicados e possíveis inconsistências.

c) Calcular estatísticas descritivas para as características numéricas, incluindo média, mediana, desvio-padrão, valores mínimos e máximos.

d) Construir visualizações para analisar as distribuições das principais características, utilizando histogramas, boxplots ou outros gráficos adequados.

e) Analisar possíveis relações entre características por meio de gráficos de dispersão e uma matriz de correlação.

f) Identificar possíveis valores discrepantes e discutir sua influência sobre os procedimentos posteriores.

g) Apresentar pelo menos três observações relevantes identificadas durante a análise exploratória.

## **4. Pré-processamento dos dados**

A equipe deverá preparar o conjunto de dados para aplicação do algoritmo K-Means.

### **Atividades obrigatórias**

a) Selecionar as características que serão utilizadas no agrupamento, justificando a escolha.

b) Tratar valores ausentes e registros inconsistentes, quando necessário.

c) Realizar a transformação de variáveis categóricas, caso sejam utilizadas.

d) Aplicar uma técnica de normalização ou padronização das características.

e) Explicar por que o pré-processamento realizado é importante para o funcionamento do K-Means.

A equipe deverá preservar uma versão dos dados antes das transformações de escala, permitindo interpretar posteriormente os agrupamentos nas unidades originais.

## **5. Aplicação do K-Means**

A equipe deverá utilizar o algoritmo K-Means para identificar agrupamentos presentes no conjunto de dados.

### **Atividades obrigatórias**

a) Executar o K-Means utilizando diferentes valores de \\(K\\), considerando pelo menos o intervalo de 2 a 10, desde que compatível com a quantidade de amostras distintas.

b) Calcular a inércia para cada valor de \\(K\\).

c) Construir o gráfico do método do cotovelo, relacionando \\(K\\) e inércia.

d) Calcular o Silhouette Score para os diferentes valores de \\(K\\).

e) Selecionar um número de clusters com base nos resultados obtidos e justificar a decisão.

f) Executar o agrupamento final utilizando o valor de \\(K\\) selecionado.

g) Apresentar a quantidade e o percentual de amostras pertencentes a cada cluster.

## **6. Interpretação dos clusters**

Após o agrupamento, a equipe deverá analisar os grupos identificados.

### **Atividades obrigatórias**

a) Calcular as médias ou medianas das principais características para cada cluster.

b) Construir uma tabela comparativa das características dos agrupamentos.

c) Produzir pelo menos duas visualizações que permitam comparar os clusters.

d) Identificar as principais características que diferenciam os grupos encontrados.

e) Atribuir uma descrição interpretativa para cada cluster, considerando o contexto do dataset.

f) Discutir se os agrupamentos representam padrões relevantes e quais limitações podem existir na interpretação dos resultados.

**Resultado esperado do Momento I:** identificação, caracterização e interpretação dos agrupamentos existentes no dataset, com justificativa quantitativa para a escolha do número de clusters.

# **Momento II – Aprendizado supervisionado: regressão**

## **7. Definição do problema de regressão**

Utilizando o mesmo dataset da primeira etapa, a equipe deverá selecionar uma característica numérica para ser estimada por meio de modelos de regressão.

### **Atividades obrigatórias**

a) Selecionar a variável dependente (alvo) e justificar sua escolha.

b) Identificar as variáveis independentes que serão utilizadas como preditoras.

c) Analisar a relação entre a variável alvo e as características selecionadas.

d) Verificar a presença de correlações elevadas entre as variáveis independentes e discutir possíveis efeitos sobre os modelos.

e) Identificar possíveis problemas de vazamento de dados, evitando utilizar variáveis que revelem diretamente o valor da característica a ser prevista.

**Importante:** A variável selecionada como alvo deverá ser excluída das características utilizadas na etapa de agrupamento, caso tenha sido originalmente considerada. A equipe deverá ajustar a análise do Momento I para garantir essa condição.

## **8. Preparação dos dados para regressão**

A equipe deverá preparar os dados para treinamento e avaliação dos modelos.

### **Atividades obrigatórias**

a) Dividir o dataset em conjuntos de treinamento e teste, utilizando uma proporção de 80% e 20%.

b) Aplicar o tratamento necessário às características categóricas e numéricas.

c) Padronizar as características preditoras, especialmente para os modelos Ridge e Lasso.

d) Garantir que as transformações sejam ajustadas somente com os dados de treinamento e posteriormente aplicadas ao conjunto de teste.

e) Utilizar a mesma divisão de dados para os três modelos, garantindo condições equivalentes de comparação.

## **9. Treinamento dos modelos de regressão**

A equipe deverá implementar e comparar três abordagens:

| Modelo           | Característica                                                          |
| ---------------- | ----------------------------------------------------------------------- |
| Regressão Linear | Regressão por mínimos quadrados ordinários, sem regularização explícita |
| Ridge            | Regressão linear com regularização L2                                   |
| Lasso            | Regressão linear com regularização L1                                   |

### **Atividades obrigatórias**

a) Treinar um modelo de regressão linear convencional.

b) Treinar um modelo Ridge, avaliando diferentes valores do hiperparâmetro \\(\alpha\\).

c) Treinar um modelo Lasso, avaliando diferentes valores do hiperparâmetro \\(\alpha\\).

d) Utilizar Grid Search com validação cruzada K-Fold para selecionar os melhores hiperparâmetros dos modelos Ridge e Lasso.

e) Registrar as configurações utilizadas nos três modelos.

f) Comparar os coeficientes obtidos pelos modelos, observando diferenças na magnitude dos pesos e possíveis coeficientes zerados.

A equipe deverá utilizar pelo menos cinco valores diferentes de \\(\alpha\\) durante a busca de hiperparâmetros dos modelos regularizados.

## **10. Avaliação e comparação dos modelos**

Os modelos deverão ser avaliados utilizando o mesmo conjunto de teste.

### **Métricas obrigatórias**

- MAE – Mean Absolute Error.
- MSE – Mean Squared Error.
- RMSE – Root Mean Squared Error.
- R² – Coeficiente de determinação.

### **Atividades obrigatórias**

a) Calcular as quatro métricas para cada modelo.

b) Construir uma tabela comparativa dos resultados, seguindo o modelo:

| Modelo           | MAE | MSE | RMSE | R² |
| ---------------- | --- | --- | ---- | -- |
| Regressão Linear |     |     |      |    |
| Ridge            |     |     |      |    |
| Lasso            |     |     |      |    |

c) Construir gráficos comparando os valores reais e previstos por cada modelo.

d) Analisar a distribuição dos resíduos dos três modelos.

e) Comparar os resultados obtidos e identificar qual modelo apresentou melhor desempenho no conjunto de teste.

f) Discutir a influência da regularização sobre o desempenho e os coeficientes dos modelos.

g) Explicar em quais situações Ridge ou Lasso podem apresentar vantagens em relação à regressão linear convencional.

h) Justificar a escolha do modelo que a equipe considera mais adequado para o problema.

## **11. Análise integrada dos resultados**

Esta etapa deverá estabelecer uma relação entre os dois momentos do trabalho.

A equipe deverá responder:

a) Quais padrões relevantes foram identificados durante o agrupamento?

b) As características que diferenciam os clusters também apresentam relação com a variável utilizada na regressão?

c) A distribuição da variável alvo apresenta diferenças entre os clusters identificados?

d) O modelo de regressão selecionado apresenta erros semelhantes nos diferentes clusters? Para responder, calcule o MAE e o RMSE separadamente para cada grupo, considerando o conjunto de teste.

e) Com base nas análises realizadas, quais conclusões podem ser obtidas sobre a estrutura do dataset e sobre a capacidade preditiva dos modelos?

Na análise por clusters, a equipe deverá utilizar os agrupamentos obtidos sem a variável alvo. As comparações devem ser interpretadas de forma exploratória, sem utilizar informações do conjunto de teste para reajustar os modelos.


## **13. Critérios de avaliação**

| Critério                                                   | Pontuação |
| ---------------------------------------------------------- | --------- |
| Análise exploratória e compreensão do dataset              | 15        |
| Pré-processamento e preparação dos dados                   | 10        |
| Aplicação do K-Means e seleção de K                        | 15        |
| Interpretação e caracterização dos clusters                | 15        |
| Implementação e configuração dos três modelos de regressão | 15        |
| Avaliação quantitativa e comparação dos modelos            | 15        |
| Análise integrada e conclusões                             | 10        |
| Organização, documentação e reprodutibilidade do notebook  | 5         |
| **Total**                                                  | **100**   |

## **14. Orientações gerais**


* As apresentações das soluções ocorrerão nos dias **26/11 e 27/11**.

* A solução deverá ser enviada por meio da **branch correspondente no GitHub**.

* A entrega deverá conter:

  * o código-fonte utilizado;
  * os gráficos, tabelas e resultados gerados durante a execução;
  * os arquivos necessários para executar a solução.

* Além do envio do código, será obrigatória a **apresentação da solução desenvolvida**.

* Não será necessária a elaboração de relatório.

O trabalho deverá apresentar não apenas a execução dos algoritmos, mas também a interpretação dos resultados e a justificativa das decisões metodológicas.

Não será suficiente apresentar somente gráficos, tabelas ou métricas sem discutir seus significados.

A utilização das bibliotecas Pandas, NumPy, Matplotlib, Seaborn e Scikit-learn é permitida.

As equipes poderão utilizar bibliotecas adicionais, desde que justifiquem sua necessidade.

Os resultados deverão ser obtidos a partir do dataset selecionado pela equipe, com código executável e análises próprias.

**Observação metodológica:** Como o trabalho integra aprendizado não supervisionado e regressão, recomendo que a escolha da variável alvo seja realizada antes da execução do K-Means. Isso permite excluir essa variável do processo de agrupamento e, posteriormente, analisar a distribuição do alvo nos clusters sem que ele tenha influenciado diretamente a formação dos grupos.
