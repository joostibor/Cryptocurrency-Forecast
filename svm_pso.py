import pandas as pd
import numpy as np

from lstm_gru import refreshExchangeRate, cryptos, visualizeAndSave
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from datetime import datetime, timedelta

#Scaling
from sklearn.preprocessing import MinMaxScaler

def dataPreprocessingAndSvmTeaching(dataframe, rbf):
    dataframe['Prediction'] = dataframe[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
    x = np.array(dataframe.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
    x = x[:len(dataframe)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
    y = np.array(dataframe['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
    y = y[:-forecasting_days] #Üres sorok kihagyása
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő
    rbf.fit(x_train, y_train) #SVM tanítása
    #Pontosság vizsgálat
    model_accuracy = rbf.score(x_test, y_test)
    print("Modell pontossága: ", model_accuracy)
    #Teszthalmazon átlagos négyzetes eltérés mérése
    predictions = rbf.predict(x_test)
    rmse = (np.sqrt(np.mean(np.square((y_test - predictions) / y_test)))) * 100
    print("Átlagos négyzetes eltérés (%): ", rmse)
    #DataFrame visszaadása
    return dataframe

forecasting_days = 30 #Előrejelzendő napok száma
#refreshExchangeRate(cryptos) #Árfolyamadatok frissítése
rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001) #Árfolyamra való tekintettel rbf függvény használata

for c in cryptos:
    df = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact.csv', index_col=False) #Adatfájl betöltése
    df = dataPreprocessingAndSvmTeaching(df, rbf)

    firsttime = True #.csv fájl import miatt létrehozott változó

    #Előrejelzés 30 napra
    for i in range(0, 30):
        if not firsttime:
            df = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_osvm.csv', index_col=False)
            df = dataPreprocessingAndSvmTeaching(df, rbf)
        last_rows = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1))[-forecasting_days:] #-->forecasting days helyett 1, és akkor 1 napot lehet előre és ezt kell majd for ciklusozni, try this shit
        forecast = rbf.predict(last_rows)
        df = df.drop(['Prediction'], axis=1)
        df.loc[len(df)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d %H:%M:%S%z') + timedelta(days=1))), forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], 0]
        df.to_csv(f'.\Exchange Rates\{c}-USD_p_osvm_test.csv', index=False)
        firsttime = False

visualizeAndSave(cryptos, ['', '-USD_p_osvm.csv'], ['', '-USD_osvm.svg'])