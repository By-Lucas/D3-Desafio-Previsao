# BIBLIOTECAS

import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import MinMaxScaler


covid_data = pd.read_csv('arquivos/time_series_covid_19_confirmed.csv')
#covid_data.head()

"""# Preparação de dados
* Como queremos prever casos para os EUA, extrairemos a linha contendo os casos de covid confirmados nos EUA.
"""

covid_data = covid_data.loc[covid_data['Country/Region'] == 'Brazil']

#covid_data

"""### Dados de pré-processamento
* Podemos descartar as colunas para 'Província/Estado', 'País/Região', 'Lat' e 'Long', SE OS  dados são apenas para os EUA e essas colunas não são necessárias para previsão.
"""

covid_data = covid_data.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

"""* Como o número de casos de Corona fica bastante grande ao longo do tempo, os cálculos do nosso modelo durante o treinamento podem ser muito lentos. Podemos corrigir isso usando o MinMaxScaler do sklearn para redimensionar nossos dados."""

scaler = MinMaxScaler()
scaler.fit(covid_data.values.T)
covid_data = scaler.transform(covid_data.values.T)

"""### Dividir em X e y
* Vamos configurar nosso X e y de tal forma que X[n] conterá os casos para uma certa quantidade de dias anteriores (time_steps) e y[n] conterá a leitura para o dia seguinte.
* Dessa forma, nosso modelo será treinado para prever o número de casos em um determinado dia com base na tendência do número de casos no número de dias de time_steps anterior.
* Após alguns testes, descobri que usar os dados dos 30 dias anteriores permitiu que nosso modelo fizesse previsões bastante precisas no 31º dia.
"""

X, y = [], []
time_steps = 30 #Quantidade de dias anteriores

for i in range(len(covid_data) - time_steps):
    x = covid_data[i:(i+time_steps), 0]
    X.append(x)
    y.append(covid_data[i+time_steps, 0])

X = np.array(X)
y = np.array(y)

"""# Particionamento de dados
* Devemos manter o conjunto de dados em ordem, pois estamos analisando uma linha do tempo cronológica dos casos de Corona, para que possamos usar os primeiros 80% dos dados como nosso treinamento e nossos testes serão os 20% restantes.
* Também precisamos remodelar as partições X[n] para que nosso modelo possa processá-las corretamente.
"""


split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

"""# Arquitetura do modelo
* Criamos nosso modelo usando uma arquitetura de rede neural recorrente.
* O modelo consiste em uma camada de entrada, seguida por três camadas LSTM que utilizam dropout para evitar que nosso modelo se ajuste demais.
* A saída é uma camada Densa com um único neurônio usando a função de ativação ReLU, pois estamos prevendo o número de casos Corona, então nossa saída será um número positivo (0, $\infty$).
"""

model = Sequential()
model.add(Input(shape=(1, time_steps)))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(48))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.compile(loss = 'mean_squared_error',
              optimizer = RMSprop(),
              metrics = ['mean_squared_error'])

model.summary()

"""# Treine o modelo
* Agora podemos treinar nosso modelo usando 20% dos dados de treinamento como nosso conjunto de validação.
* O modelo usará o ReduceLROnPlateau para diminuir nossa taxa de aprendizado sempre que nossos platôs MSE de validação por três épocas para melhor precisão.
"""