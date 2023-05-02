import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, datetime

def saveNewPredictsToCsv(predictions):
    df = pd.read_csv('BTC-USD.csv')
    for i in range(0, len(predictions)):
        df.loc[len(df.index)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d') + timedelta(days=1)).date()), predictions[i][1], predictions[i][1], predictions[i][1], predictions[i][1], predictions[i][1], 0]
    df.to_csv('BTC-USD_2.csv', index=False)

#Modell betöltése
model = load_model('btc_model_wo_fdlogic.h5')

#Adatok betöltése
data = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=['Date'])

#Adatok előkészítése
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#Időintervallumok definiálása
timestamps = 30 

#Modell tesztelése a 2020 utáni adatokkal
test_data = data['2020':]
fact_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - timestamps:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

predictions = []
for i in range (1,11):
    real_data = [model_inputs[len(model_inputs) + i - timestamps:len(model_inputs) + i, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict([real_data, real_data])
    prediction = scaler.inverse_transform(prediction)
    predictions.append((i, prediction[0][0]))
saveNewPredictsToCsv(predictions)