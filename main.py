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
covid_data.head()