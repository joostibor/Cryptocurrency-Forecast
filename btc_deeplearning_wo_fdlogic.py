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

#Tanító tömbök létrehozása
x_train, y_train = [], []

for x in range(timestamps, len(scaled_data)):
    x_train.append(scaled_data[x-timestamps:x, 0])
    y_train.append(scaled_data[x, 0])

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
merged_model.save('btc_model_wo_fdlogic.h5')

"""

#Modell tesztelése a 2020 utáni adatokkal
test_data = data['2020':]
fact_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - timestamps:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
for x in range(timestamps, len(model_inputs)):
    x_test.append(model_inputs[x-timestamps:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#x_test intervallumon árak előrejelzése
predicted_prices = merged_model.predict([x_test, x_test])
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(fact_prices, color='black', label='Valós napi záróárfolyam')
plt.plot(predicted_prices, color='green', label='Modell által előrejelzett árfolyam')
plt.title('BTC Árfolyam előrejelzés')
plt.xlabel('Idő')
plt.ylabel('Ár')
plt.legend(loc='upper left')
plt.show()

#Következő napi árfolyam előrejelzése
real_data = [model_inputs[len(model_inputs) + 1 - timestamps:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = merged_model.predict([real_data, real_data])
prediction = scaler.inverse_transform(prediction)
print(prediction)
"""