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

