

## Orientações gerais

* As apresentações das soluções ocorrerão nos dias **08/10 e 09/10**.

* A solução deverá ser enviada por meio da **branch correspondente no GitHub**.

* A entrega deverá conter:

  * o código-fonte utilizado;
  * os gráficos, tabelas e resultados gerados durante a execução;
  * os arquivos necessários para executar a solução.

* Além do envio do código, será obrigatória a **apresentação da solução desenvolvida**.

* Não será necessária a elaboração de relatório.

* Durante a apresentação, o estudante deverá explicar:

  * a análise e a preparação do dataset;
  * as inconsistências encontradas e as decisões tomadas;
  * as técnicas de normalização avaliadas;
  * a comparação dos resultados;
  * o processo de escolha do melhor valor de \(K\);
  * as principais partes do código.

* A avaliação considerará conjuntamente:

  * as atividades realizadas em casa;
  * a prova escrita;
  * a apresentação da solução computacional.

* Não serão aceitas apresentações após o dia **09/10**.

* Caso algum estudante ou equipe necessite apresentar antes dessas datas, poderá solicitar antecipadamente o agendamento.


## Questão 1 

Realize uma análise exploratória do conjunto de dados disponibilizado e prepare os dados para a etapa de classificação.

Sua análise deve contemplar, pelo menos:

1. Identificação das dimensões do dataset, dos tipos de dados e da variável que será utilizada como classe.

2. Verificação da existência de inconsistências:

3. Apresentação de uma análise estatística das variáveis numéricas:

4. Análise da distribuição das classes.

5. Avaliação das escalas das variáveis numéricas. A partir dos resultados encontrados, discuta se a normalização ou padronização é necessária para a aplicação do KNN.

6. Aplicação das transformações consideradas necessárias para preparar o dataset para a etapa de classificação.

Apresente:

* os principais resultados obtidos;
* a justificativa para cada coluna ou linha removida;
* o tratamento aplicado às inconsistências;
* a justificativa para normalizar, padronizar ou manter os dados em sua escala original;
* as dimensões do dataset antes e depois do tratamento;
* uma versão final do dataset preparada para a Questão 2.

Todas as decisões devem ser justificadas com base nas características observadas nos dados e no funcionamento do KNN. A simples execução de funções, sem interpretação dos resultados, não será considerada uma resposta completa.

## Questão 2 

Utilize o dataset preparado na Questão 1 para desenvolver um modelo de classificação baseado no algoritmo K-Nearest Neighbors.

O experimento deverá comparar o desempenho do KNN em quatro configurações:

1. Dados sem normalização ou padronização;
2. Dados transformados pela primeira técnica escolhida;
3. Dados transformados pela segunda técnica escolhida;
4. Dados transformados pela terceira técnica escolhida.

As três técnicas de transformação são de livre escolha. 

Organize os resultados em uma tabela semelhante à seguinte:

| Configuração | Técnica utilizada | Acurácia |
| ------------ | ----------------- | -------: |
| 1            | Sem normalização  |          |
| 2            | Técnica 1         |          |
| 3            | Técnica 2         |          |
| 4            | Técnica 3         |          |

### Análise dos resultados

Com base nos resultados obtidos, responda:

1. Qual configuração apresentou a maior acurácia?
2. A normalização melhorou o desempenho do KNN? Justifique numericamente.
3. Qual técnica de normalização foi mais adequada para esse dataset?
4. As características estatísticas identificadas na Questão 1 ajudam a explicar os resultados?
5. Existe evidência de que o dataset pode ser utilizado sem normalização? Justifique.
6. A diferença entre as acurácias é suficiente para afirmar que uma configuração é claramente superior às demais?
7. Considerando o funcionamento do KNN e o cálculo de distâncias entre as amostras, explique por que as técnicas de normalização podem produzir resultados diferentes.

Apresente : tabela comparativa e uma conclusão indicando qual configuração seria adotada como solução final. Todas as configurações devem ser avaliadas sob as mesmas condições experimentais.



## Questão 3 

Utilize a configuração de normalização considerada mais adequada na Questão 2 para investigar qual valor de \(K\) produz o melhor desempenho no algoritmo K-Nearest Neighbors.

### Procedimentos obrigatórios

1. Avalie diferentes valores de \(K\), utilizando pelo menos 15 valores distintos.
2. Treine e avalie o KNN para cada valor de \(K\).
3. Utilize as mesmas variáveis, técnica de normalização, métrica de distância e condições experimentais em todos os testes.
4. Calcule a acurácia de treinamento e a acurácia de validação para cada valor de \(K\).
5. Organize os resultados em uma tabela contendo:

| Valor de K | Acurácia de treinamento | Acurácia de validação |
| ---------: | ----------------------: | --------------------: |
|          1 |                         |                       |
|          2 |                         |                       |
|        ... |                         |                       |

6. Construa um gráfico de linhas contendo:

   * valores de \(K\) no eixo horizontal;
   * acurácia no eixo vertical;
   * uma linha para a acurácia de treinamento;
   * uma linha para a acurácia de validação;
   * legenda, título e identificação dos eixos.

### Análise dos resultados

Com base na tabela e no gráfico, responda:

1. Qual valor de \(K\) apresentou a maior acurácia de validação?
2. Esse valor deve ser escolhido automaticamente ou existem outros valores com resultados semelhantes?
3. O que acontece com o modelo quando \(K\) é muito pequeno?
4. O que acontece com o modelo quando \(K\) é muito grande?
5. O gráfico apresenta indícios de sobreajuste ou subajuste? Justifique.
6. Qual valor de \(K\) você considera mais adequado para o problema? Justifique sua escolha considerando tanto a acurácia quanto a capacidade de generalização do modelo.

Após selecionar o melhor valor de \(K\), treine o modelo final utilizando a configuração escolhida e apresente sua acurácia no conjunto de teste.

A escolha do valor de \(K\) deverá ser realizada com os dados de validação ou por validação cruzada. O conjunto de teste não deverá ser utilizado para escolher \(K\), pois deverá permanecer reservado para a avaliação final do modelo.



# Duplas
Levi de Sousa Nascimento 
Kemilly da Conceição Mota
Maria Laísa Viana Gomes
dataset: https://archive.ics.uci.edu/dataset/45/heart%2Bdisease


Francisco Emanuel de Sousa Oliveira
Francisco Kauan Lima Moura
dataset: https://archive.ics.uci.edu/dataset/571/hcv+data

Camila Araújo
Yasmin Coelho
dataset: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

João Pedro 
Thiago Pinto Gomes
dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Joao victor 
Antonio erick
dataset: https://archive.ics.uci.edu/dataset/186/wine+quality

João Gilberto
Marcone Silva
dataset: https://archive.ics.uci.edu/dataset/863/maternal+health+risk.

Gabriely Moura Altenhofen
Habacuqui
dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients