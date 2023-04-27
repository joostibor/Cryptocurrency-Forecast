import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM, GRU, concatenate
from keras.models import Sequential

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

#Neurális hálózat építése, tanítása és mentése
#LSTM modell
lstm_model = Sequential()
lstm_model.add(LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
#model.add(LSTM(units=50, return_sequences=True))
#model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=True))
#model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

#GRU modell
gru_model = Sequential()
gru_model.add(GRU(units=30, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh'))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(units=1))

#Összevont modell
merged_model = Sequential()
merge = concatenate(lstm_model, gru_model) #itt a hiba
merged_model.add(merge)
merged_model.add(Dense(units=1, activation="relu"))
merged_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
merged_model.fit(x_train, y_train, epochs=100, batch_size=32)

merged_model.save('btc_model_w_fdlogic.h5')

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
predicted_prices = merged_model.predict(x_test)
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

prediction = merged_model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)