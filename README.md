# Desafio D3, previsão de casos Corona Virus


<p align="center">
    <img src="gif.gif" width=600px>
</p><br>

# Todas as informações
## Carregar dados
* Busque dados do arquivo csv e coloque-os em um Pandas DataFrame.
import pandas as pd

covid_data = pd.read_csv('arquivos/time_series_covid_19_confirmed.csv')
covid_data.head()
## Preparação de dados
* Como queremos prever casos para os BRasil, extrairemos a linha contendo os casos de covid confirmados nos Brasil.<br><br>

us_covid_data = covid_data.loc[covid_data['Country/Region'] == 'Brazil']<br>

display(us_covid_data)<br><br>
## Dados de pré-processamento
* Podemos descartar as colunas para 'Província/Estado', 'País/Região', 'Lat' e 'Long', os dados são apenas para o Brasil e essas colunas não são necessárias para previsão.

us_covid_data = us_covid_data.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])<br>

* Como o número de casos de Corona fica bastante grande ao longo do <br>tempo, os cálculos do nosso modelo durante o treinamento podem ser <br>muito lentos. Podemos corrigir isso usando o MinMaxScaler do sklearn <br>para redimensionar nossos dados.<br>
from sklearn.preprocessing import MinMaxScaler<br>

scaler = MinMaxScaler()<br>
scaler.fit(us_covid_data.values.T)<br>
us_covid_data = scaler.transform(us_covid_data.values.T)<br><br>

## Dividir em X e y
* Vamos configurar nosso X e y de tal forma que X[n] conterá os casos para uma certa quantidade de dias anteriores (time_steps) e y[n] conterá a leitura para o dia seguinte.
* Dessa forma, nosso modelo será treinado para prever o número de casos em um determinado dia com base na tendência do número de casos no número de dias de time_steps anterior.
* Após alguns testes, descobri que usar os dados dos 30 dias anteriores permitiu que nosso modelo fizesse previsões bastante precisas no 31º dia.


import numpy as np<br>

X, y = [], []<br>
time_steps = 30<br>

for i in range(len(us_covid_data) - time_steps):<br>
    x = us_covid_data[i:(i+time_steps), 0]<br>
    X.append(x)<br>
    y.append(us_covid_data[i+time_steps, 0])<br>

X = np.array(X)<br>
y = np.array(y)<br>

## Particionamento de dados
* Devemos manter o conjunto de dados em ordem, pois estamos analisando uma linha do tempo cronológica dos casos de Corona, para que possamos usar os primeiros 80% dos dados como nosso treinamento e nossos testes serão os 20% restantes.
* Também precisamos remodelar as partições X para que nosso modelo possa processá-las corretamente.


split = int(len(X) * 0.8)<br>

X_train = X[:split]<br>
X_test = X[split:]<br>
y_train = y[:split]<br>
y_test = y[split:]<br>

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))<br>
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))<br>

## Arquitetura do modelo
* Criamos nosso modelo usando uma arquitetura de rede neural recorrente.
* O modelo consiste em uma camada de entrada, seguida por três camadas LSTM que utilizam dropout para evitar que nosso modelo se ajuste demais.
* A saída é uma camada Densa com um único neurônio usando a função de ativação ReLU


import tensorflow as tf<br>
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM<br>
from tensorflow.keras.models import Model<br>
from tensorflow.keras.models import Sequential<br>
from tensorflow.keras.optimizers import RMSprop<br>

model = Sequential()<br>
model.add(Input(shape=(1, time_steps)))<br>
model.add(LSTM(48, return_sequences=True))<br>
model.add(Dropout(0.4))<br>
model.add(LSTM(48, return_sequences=True))<br>
model.add(Dropout(0.2))<br>
model.add(LSTM(48))<br>
model.add(Dropout(0.2))<br>
model.add(Dense(1, activation='relu'))<br>


model.compile(loss = 'mean_squared_error',
              optimizer = RMSprop(),
              metrics = ['mean_squared_error'])<br>

model.summary()


## Treinando  nosso modelo
* Agora podemos treinar nosso modelo usando 20% dos dados de treinamento como nosso conjunto de validação.
* O modelo usará o ReduceLROnPlateau para diminuir nossa taxa de aprendizado sempre que nossos platôs MSE de validação por três épocas para melhor precisão.
<br>

from keras.callbacks import ReduceLROnPlateau

batchsize = 100<br>
epochs =  100<br>
<br>
learning_rate_reduction = ReduceLROnPlateau(monitor='val_mean_squared_error', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=1e-10)<br>

history = model.fit(X_train,
                    y_train,
                    batch_size=batchsize,
                    epochs=epochs,
                    validation_split=0.2,
                    shuffle=False,
                    callbacks=[learning_rate_reduction])<br><br>

* Plote os valores de perda e MSE do modelo ao longo do treinamento.<br>

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])<br>
plt.plot(history.history['val_loss'])<br>
plt.title('Modelo de percas')<br>
plt.ylabel('Percas')<br>
plt.xlabel('Épocas')<br>
plt.legend(['Treino', 'Valor'])<br>
plt.show()<br>

plt.plot(history.history['mean_squared_error'])<br>
plt.plot(history.history['val_mean_squared_error'])<br>
plt.title('Erro de modelo')<br>
plt.ylabel('Erro médio')<br>
plt.xlabel('Épocas')<br>
plt.legend(['Treino', 'Valor'])<br>
plt.show()<br>

# Previsões do modelo
* Para ver a precisão do nosso modelo, primeiro o usamos para prever a saída de nossos dados X_test.
* Em seguida, redimensionamos nossos dados de previsão e y_test de volta aos limites originais do conjunto de dados para plotar com precisão seus valores.
* Por fim, podemos traçar os casos reais de Covid em comparação com nossos casos previstos de Covid para ver a precisão geral do nosso modelo.


y_pred = model.predict(X_test)<br>
y_pred = scaler.inverse_transform(y_pred)<br>
y_test = scaler.inverse_transform(y_test.reshape(-1,1))<br>

plt.plot(y_pred, color='red')<br>
plt.plot(y_test, color='blue')<br>
plt.title('Casos de Covid reais x previstos (dados de teste)')<br>
plt.ylabel('Número de casos')<br>
plt.xlabel('Dia')<br>
plt.legend(['Previsto', 'Atual'])