import pandas as pd
import numpy as np
import subprocess

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, datetime

def saveNewPredictsToCsv(predictions):
    df = pd.read_csv('BTC-USD_predict_wfdl.csv')
    for i in range(0, len(predictions)):
        df.loc[len(df.index)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d') + timedelta(days=1)).date()), predictions[i][1], predictions[i][1], predictions[i][1], predictions[i][1], 0]
    df.to_csv('BTC-USD_predict_wfdl.csv', index=False)

def modelLearnAgain():
    subprocess.run(["python", "btc_deeplearning_w_fdlogic.py"])

timestamps = 30 
#100 nap előrejelzése, 10 naponként modell újratanítása
for i in range (1,101,10):
    model = load_model('.\Models\\btc_model_wo_fdlogic.h5')
    data = pd.read_csv('BTC-USD_predict_wfdl.csv', index_col='Date', parse_dates=['Date'])
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    test_data = data['2020':]
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - timestamps:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    predictions = []
    for j in range(1,11):
        real_data = [model_inputs[len(model_inputs) + j - timestamps:len(model_inputs) + j, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = model.predict([real_data, real_data])
        prediction = scaler.inverse_transform(prediction)
        predictions.append((j, prediction[0][0]))
    saveNewPredictsToCsv(predictions)
    modelLearnAgain()