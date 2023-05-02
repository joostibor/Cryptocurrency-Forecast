import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Dropout, LSTM, GRU, concatenate
from keras.models import Model

#Adat beolvasása
data = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=['Date'])

#Adatok előkészítése
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#Időintervallumok definiálása
timestamps = 30 
future_days = 30

#Tanító tömbök létrehozása
x_train, y_train = [], []

for x in range(timestamps, len(scaled_data)-future_days):
    x_train.append(scaled_data[x-timestamps:x, 0])
    y_train.append(scaled_data[x+future_days, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#LSTM oldal
lstm_input = Input(shape=(x_train.shape[1], 1), name='LSTM input')
lstm_1 = LSTM(units=30, return_sequences=True, name='First_LSTM')(lstm_input)
lstm_2 = Dropout(0.2, name='first_LSTM_Droupout')(lstm_1)
lstm_3 = LSTM(units=50, name='Second_LSTM')(lstm_2)
lstm_4 = Dropout(0.2, name='second_LSTM_Droupout')(lstm_3)
lstm_5 = Dense(units=1, name='LSTM_output_Dense')(lstm_4)

#GRU oldal
gru_input = Input(shape=(x_train.shape[1], 1), name='GRU_input')
gru_1 = GRU(units=30, activation='tanh', name='first_GRU')(gru_input)
gru_2 = Dropout(0.2, name='GRU_dropout')(gru_1)
gru_3 = Dense(units=1, name='GRU_output_Dense')(gru_2)

#Összevonás
lstm_gru = concatenate([lstm_5, gru_3], name='concatanated_layer')

#Végső kimenet
output = Dense(units=1, activation="relu", name='Output_Dense_layer')(lstm_gru)

#Modell tanítása és mentése
merged_model = Model(inputs=[(lstm_input, gru_input)], outputs=[output], name='Merged_model')
merged_model.compile(optimizer='adam', loss='mean_squared_error')
merged_model.fit([x_train, x_train], y_train, epochs=25, batch_size=32)
merged_model.save('btc_model_w_fdlogic.h5')