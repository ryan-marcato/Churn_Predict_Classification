# Churn_Predict_Classification
- Dataset retirado do Kaggle link: https://www.kaggle.com/datasets/shubh0799/churn-modelling
- Colunas Renomeadas para o portugues.

# Objetivo
- Realizar previsões da classificação de churn.
- Modelo busca priorizar pessoas com alta probabilidade de saida.

# Projeto
Este projeto de Machine Learning é dividido em duas partes principais: Pré-processamento de Dados e Treinamento de Modelos.

## Pré-requisitos
Certifique-se de ter Python 3.12 instalado e as seguintes bibliotecas:
- ```pandas```
- ```matplotlib```
- ```scikit-plot```
- ```scikit-learn```
- ```scikit-optmize```
- ```xgboost```
- ```feature-engine```

## Instalação das Dependências
Para instalar as bibliotecas necessárias, execute:
```bash
pip install pandas matplotlib scikit-plot scikit-learn scikit-optimize xgboost feature-engine
```

## Estrutura do Projeto
1. ### Pré-processamento de Dados
No notebook de pré-processamento, são realizadas as etapas de limpeza, preparação e codificação dos dados.
### Instruções para o Pré-processamento
1. #### Inicie o Jupyter Notebook:
```bash
jupyter notebook
```
2. Abra o Notebook de Pré-processamento e execute as células conforme instruções.
##### Importar Bibliotecas
Comece importando as blibliotecas:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.encoding import OneHotEncoder
```
##### Carregar e Dividir os Dados
Carrregue o conjunto de dados e divida-o em treino e teste:
```python
df = pd.read_csv("seu_arquivo.csv")  # Substitua pelo caminho do arquivo
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
##### Codificar Variáveis Categóricas
Utilize o ```OneHotEncoder``` para variáveis categóricas:
```python
encoder = OneHotEncoder(variables=['sua_variavel_categorica'])  # Substitua pelo nome da variável
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```

2. ### Treinamento de Modelos
O notebook de treinamento contém a configuração, o treinamento e a avaliação dos modelos. 
### Instruções para Treinamento
1. #### Inicie o Jupyter Notebook:
```bash
jupyter notebook
```
2. Abra o Notebook de Treinamento e execute as células conforme instruções.
##### Importar Bibliotecas
No início, importe as bibliotecas:
```python
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt  # type: ignore
from sklearn import metrics
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
```
##### Configurar e Treinar o Modelo
Com ```BayesSearchCV```, faça a otimização dos hiperparâmetros dos modelos ```RandomForestClassifier``` e ```XGBClassifier```:
```python
space = {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(1e-6, 0.1, prior="log-uniform"),
    'gamma': Real(1e-6, 0.1, prior="log-uniform"),
    'min_child_weight': Integer(1, 10),
    'scale_pos_weight': Real(1e-6, 50, prior="log-uniform"),
    'max_depth': Integer(2, 12),
    'subsample': Real(0.6, 0.8),
    'colsample_bytree': Real(0.6, 0.8),
    'reg_alpha': Real(1e-6, 0.1, prior="log-uniform"), 
    'reg_lambda': Real(1e-6, 0.1, prior="log-uniform"),
    'objective': Categorical(['binary:logistic']),
}

opt = BayesSearchCV(xbg,
                    space,
                    n_iter=30,
                    random_state=42)
```
##### Avaliar o Modelo
Utilize ```metrics``` e ```scikit-plot``` para avaliação:
```python
predict_train = opt.predict(X_train)
predict_test = opt.predict(X_test)
proba_train = opt.predict_proba(X_train)
proba_test = opt.predict_proba(X_test)
```

```python
train_acc = metrics.accuracy_score(y_train, predict_train)
test_acc = metrics.accuracy_score(y_test, predict_test)
train_roc = metrics.roc_auc_score(y_train, proba_train[:, 1])
test_roc = metrics.roc_auc_score(y_test, proba_test[:, 1])
```

```python
print("Acc train:", train_acc)
print("Acc test:", test_acc)
print("Roc auc score:", train_roc)
print("Roc auc score:", test_roc)
```

```python
skplt.metrics.plot_lift_curve(y_test, proba_test)
plt.show()
```

### Observações
- Substitua ```"seu_arquivo.csv"``` e ```'sua_variavel_categorica'``` com os nomes do seu arquivo e variável específica.
- Adapte o pré-processamento e o treinamento conforme as necessidades do seu dataset.
