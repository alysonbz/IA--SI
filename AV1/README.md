# AVALIAÇÃO 1
> Orientações para execução da prova.

PRAZO PARA ENVIO E APRESENTAÇÃO: 14/01/2025

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. Leia antentemente antes. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

Adriel Bastos e Aryely  Matos: 

clssificação: https://www.kaggle.com/datasets/erdemtaha/cancer-data

regressão: https://www.kaggle.com/datasets/abrambeyer/openintro-possum

Ingrid Melo e Helen Brandão: 

classificação: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

regressão: https://www.kaggle.com/datasets/mirichoi0218/insurance

Eryka Rodrigues e Gabriel Broa : 

classificação: https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset

regressão: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

Francisco de Paulo e Natércia Souza: 

classificação: https://www.kaggle.com/datasets/whenamancodes/predict-diabities

regressão: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

Willian Mota e Ruan de Lima: 

classificação: https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings

regressão: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

Glauber Melo e Emily Hannah : 

classificação: https://www.kaggle.com/datasets/prathamtripathi/drug-classification

regressão: https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset

Antônio Kauê e José Basílio: 

classificação :https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

regressão: https://www.kaggle.com/datasets/kapturovalexander/hp-lenovo-acer-asus-samsung-companies-share-prices

Tauã Lima e Vitor Castro: 

calssificação: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

regressão: https://www.kaggle.com/datasets/kapturovalexander/ferrari-and-tesla-share-prices-2015-2023

Mizael Gomes e Vitor Emanuel:

classificação: https://www.kaggle.com/datasets/praveengovi/credit-risk-classification-dataset?select=customer_data.csv

regressão: https://www.kaggle.com/datasets/kapturovalexander/activision-nintendo-ubisoft-ea-stock-prices

Italo Cesar e Ruan Frota: 

classificação: https://www.kaggle.com/datasets/mssmartypants/water-quality

regressão: https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge

Raimundo Rafael e Ryan Guilherme:

classificação: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

regressão: https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset

### Questão 1

```questao1.py```

Neste primeiro exercício você deve realizar manipulação em um dataset com a biblioteca pandas e realizar o pré-processamento deste.

#### Instruções:

1) Importe as bibliotecas necessárias.
   
2) Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.

3) Verifique quais colunas são as mais relevantes e crie um novo dataframe com somente as colunas necesárias. 
    
4) Print o dataframe final e mostre a distribuição de classes que você deve classificar

5) Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

6) Há necessidade de mais algum pré-processamento ?

7) Salve o dataset atualizado se houver modificações. Faça uma renomeação para ``nome_do_dataset_ajustado.csv``

### Questão 2

```questao2.py```

Neste segundo exercício você deve realizar uma classificação utilizando KNN implementado de forma manual.

#### Instruções 

1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Sem normalizar o conjunto de dados divida o dataset em treino e teste.
4) Implemente o Knn exbindo sua acurácia nos dados de teste, utilize um valor de k = 7
5) Compare as acurácias considerando que 4 possíveis cálculos de distâncias diferentes:
   a) distância de mahalanobis.
   b) distancia de chebyshev
   c) distância de manhattan
   d) distancia euclidiana


### Questão 3

```questao3.py```

Considerando o a melhor distancia observada no exercício anterior, neste você deve verificar se a normalização interfere nos resultados de sua classificação.

#### Instruções

1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Normalize o conjunto de dados com normalização logarítmica  e verifique a acurácia do knn.
4) Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
5) Print as duas acuracias lado a lado para comparar. 


### Questão 4

```questao4.py```

Com base nas paramentrizações vistas anteriormento, neste exercicio você deve buscar saber a melhor parametrização do knn implementado na questão anterior.

#### Instruções

1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Normalize com a melhor normalização o conjunto de dados se houver melhoria.
4) Plote o gráfico com o a indicação do melhor k.


### Questão 5

```questao5.py```

Observe o dataset de regressão e realize o pré-processamento. Verifique qual atributo será o alvo para regressão no seu dataset
e faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
Lembre de comprovar via gráfico. Caso necessário remova colunas que são insignificantes, valore NaN também devem ser removidos.

### Questão 6

```questao6.py```

Utilizando o atributo mais relevante calculado na questão 5, implemente uma regressão linear utilizando somente este atributo mais
relevante, para predição do atributo alvo determinado na questão 5 também. Mostre o gráfico da reta de regressão  em conjunto com a nuvem 
de atributo. Determine também os valores: 
RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.


#### Instruções

1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Normalize com a melhor normalização o conjunto de dados se houver melhoria.
4) Plote o gráfico com o a indicação do melhor k.


### Questão 7

```questao7.py```

Implemente um modelo de regressão linear para resolver o problema a seguir. Siga os passos abaixo:  

**Divisão de Dados:** Implemente manualmente o método *k-fold* e a validação cruzada (*cross-validation*).  

**Modelos a Serem Testados:** Realize a regressão linear clássica, Ridge e Lasso.  

**Métricas de Avaliação:** Implemente manualmente as seguintes métricas e calcule-as para cada modelo:  
   - RSS (*Residual Sum of Squares*);  
   - MSE (*Mean Squared Error*);  
   - RMSE (*Root Mean Squared Error*);  
   - R^2 (*Coeficiente de Determinação*).  

**Análise de Desempenho:** Compare os resultados das métricas para os três tipos de regressão linear e identifique qual deles apresentou o melhor desempenho.  

Obs.: Lembre de documentar as etapas e justifique a escolha do modelo com melhor desempenho.


### Observações para o Relatório - PRAZO PARA ENVIO E APRESENTAÇÃO: 14/01/2025

Discutir **organizadamente** na sessão de resultados os números obtidos de cada questão.
Ao concluir o relatório, incluir na pasta AV1 de sua branch um aquivo em forma de PDF.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento.  Leia atentamente antes.
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true
