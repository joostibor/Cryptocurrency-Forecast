import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Dropout, LSTM, GRU, concatenate
from keras.models import Model, load_model
from datetime import timedelta, datetime

def refreshExchangeRate(cryptocurrencies):
    for c in cryptocurrencies:
        crypto_exc = yf.Ticker(f'{c}-USD')
        hist = crypto_exc.history(start="2009-01-03", end=dt.datetime.now()) #Kezdődátum megegyezik a Bitcoin indulási idejével
        hist.drop(hist.columns[[5, 6]], axis=1, inplace=True) #Felesleges oszlopok törlése
        hist.to_csv(f'.\Exchange Rates\{c}-USD_fact.csv', sep=',',index=True)

def makeModelLSTMGRUWoFdlogic(cryptocurrencies, firsttime):
    for c in cryptocurrencies:
        #Adatfájl betöltése
        if firsttime:
            data = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact.csv', index_col='Date', parse_dates=['Date'])
        else:
            data = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_wofdl.csv', index_col='Date', parse_dates=['Date'])
        
        #Scaling
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        
        #Időablak definiálása
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
        merged_model.fit([x_train, x_train], y_train, epochs=10, batch_size=32)
        merged_model.save(f'.\Models\{c}_model_wo_fdlogic.h5')

def saveNewPredictsToCsv(predictions, crypto):
    firsttime = True
    if firsttime:
        df = pd.read_csv(f'.\Exchange Rates\{crypto}-USD_fact.csv')
    else:
        df = pd.read_csv(f'.\Exchange Rates\{crypto}-USD_p_wofdl.csv')
    for i in range(0, len(predictions)):
        df.loc[len(df.index)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d %H:%M:%S%z') + timedelta(days=1))), predictions[i][1], predictions[i][1], predictions[i][1], predictions[i][1], 0]
    df.to_csv(f'.\Exchange Rates\{crypto}-USD_p_wofdl.csv', index=False)

def testModelAndFutureRatePredict(cryptocurrencies, nametags):
    timestamps = 30 
    firsttime = True

    for c in cryptocurrencies:
        #Modell tesztelése
        data = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact.csv', index_col='Date', parse_dates=['Date'])
        scaler = MinMaxScaler(feature_range=(0,1))
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
        model = load_model(f'.\Models\{c}{nametags[0]}')
        predicted_prices = model.predict([x_test, x_test])
        predicted_prices = scaler.inverse_transform(predicted_prices)
        plt.clf()
        plt.plot(fact_prices, color='black', label='Valós napi záróárfolyam')
        plt.plot(predicted_prices, color='green', label='Modell által előrejelzett árfolyam')
        plt.title(f'LSTM-GRU hibrid modell árfolyami előrejelzési tesztje {c} kriptovalután, 2020 utáni adatokkal')
        plt.xticks([])
        plt.ylabel('Ár ($)')
        plt.legend(loc='upper left')
        plt.savefig(f'.\SVG Format Figures\{c}-USD_wofdl_test.svg', format='svg')

        #50 nap előrejelzése, 10 naponként modell újratanítása     
        for i in range (1,50,10):
            if not firsttime:
                data = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_wofdl.csv', index_col='Date', parse_dates=['Date'])
            predictions = []
            for j in range(1,11):
                real_data = [model_inputs[len(model_inputs) + j - timestamps:len(model_inputs) + j, 0]]
                real_data = np.array(real_data)
                real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

                prediction = model.predict([real_data, real_data])
                prediction = scaler.inverse_transform(prediction)
                predictions.append((j, prediction[0][0]))
            saveNewPredictsToCsv(predictions, c)
            firsttime = False
            makeModelLSTMGRUWoFdlogic([c],False)

def visualizeAndSave(cryptocurrencies):
    for c in cryptocurrencies:
        p_data = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_wofdl.csv', index_col='Date', parse_dates=['Date'])
        f_data = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact.csv', index_col='Date', parse_dates=['Date'])

        plt.clf()
        plt.plot(p_data['2023':]['Close'], color='green', label='Modell alapján előrejelzett napi záró árfolyam')
        plt.plot(f_data['2023':]['Close'], color='black', label='Valós napi záró árfolyamok')
        plt.title(f'{c} záró árfolyam előrejelzés LSTM-GRU hibrid modellel')
        plt.xticks(rotation=45, fontsize=8)
        plt.ylabel('Ár ($)')
        plt.legend(loc='upper left')
        plt.savefig(f'.\SVG Format Figures\{c}-USD_wofdl.svg', format='svg')
        plt.show()

cryptos = ['BTC', 'ETH', 'BNB', 'DOGE', 'LTC'] #Bitcoin, Ethereum, Binance, DogeCoin, Litecoin
model_filenametags = ['_model_wo_fdlogic.h5']
refreshExchangeRate(cryptos) #Árfolyamadatok aktualizálása a futtatás napjáig
makeModelLSTMGRUWoFdlogic(cryptos, True) #LSTM-GRU hibrid modellek elkészítése
testModelAndFutureRatePredict(cryptos, model_filenametags) #Árfolyam előrejelzés
visualizeAndSave(cryptos) #Megjelenítés és grafikon mentése