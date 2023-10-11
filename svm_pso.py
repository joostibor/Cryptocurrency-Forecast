import pandas as pd
import numpy as np

from lstm_gru import refreshExchangeRate, cryptos, visualizeAndSave
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from datetime import datetime, timedelta

#cryptos = ['BTC'] #Előrejelzendő kriptovaluták
forecasting_days = 30 #Előrejelzendő napok száma
refreshExchangeRate(cryptos) #Árfolyamadatok frissítése
df = pd.read_csv('.\Exchange Rates\BTC-USD_fact.csv', index_col=False) #Adatfájl betöltése
df['Prediction'] = df[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
x = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
x = x[:len(df)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
y = np.array(df['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
y = y[:-forecasting_days] #Üres sorok kihagyása
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő

#SVM elkészítése
rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001) #Árfolyamra való tekintettel rbf függvény használata
rbf.fit(x_train, y_train) #SVM tanítása
model_accuracy = rbf.score(x_test, y_test)
print("Modell pontossága: ", model_accuracy)

#SVM tesztelése és összehasonlítása a tényadatokkal
print("----------Előrejelzés-Teszt fact adatok----------")
predictions = rbf.predict(x_test)
for i in range(0, len(predictions)):
    print(f'{predictions[i]},{y_test[i]}')

firsttime = True

#Előrejelzés 30 napra
for i in range(0, 30):
    if not firsttime:
        df = pd.read_csv('.\Exchange Rates\BTC-USD_p_osvm.csv', index_col=False)
        df['Prediction'] = df[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
        x = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
        x = x[:len(df)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
        y = np.array(df['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
        y = y[:-forecasting_days] #Üres sorok kihagyása
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő
        rbf.fit(x_train, y_train) #SVM tanítása
    last_rows = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1))[-forecasting_days:] #-->forecasting days helyett 1, és akkor 1 napot lehet előre és ezt kell majd for ciklusozni, try this shit
    forecast = rbf.predict(last_rows)
    df = df.drop(['Prediction'], axis=1)
    df.loc[len(df)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d %H:%M:%S%z') + timedelta(days=1))), forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], 0]
    df.to_csv('.\Exchange Rates\BTC-USD_p_osvm.csv', index=False)
    firsttime = False

#Több kriptos ciklus kell + a grafikonok