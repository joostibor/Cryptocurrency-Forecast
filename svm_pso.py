import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np

from lstm_gru import refreshExchangeRate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

cryptos = ['BTC'] #Előrejelzendő kriptovaluták
forecasting_days = 30 #Előrejelzendő napok száma
#refreshExchangeRate(cryptos) #Árfolyamadatok frissítése
df = pd.read_csv('.\Exchange Rates\BTC-USD_fact.csv') #Adatfájl betöltése
df.drop(['Date', 'Open', 'High', 'Low', 'Volume'], axis=1, inplace=True) #Felesleges oszlopok törlése, mérvadó adatsorként a záró árfolyamot tartalmazó oszlop marad
df['Prediction'] = df[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
x = np.array(df.drop(['Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
x = x[:len(df)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
y = np.array(df['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
y = y[:-forecasting_days] #Üres sorok kihagyása
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő

#SVM elkészítése
rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001) #Árfolyamra való tekintettel rbf függvény használata
rbf.fit(x_train, y_train) #SVM tanítása
model_accuracy = rbf.score(x_test, y_test)
print(model_accuracy)

#SVM tesztelése és összehasonlítása a tényadatokkal
predictions = rbf.predict(x_test)
for i in range(0, len(predictions)):
    print(f'{predictions[i]},{y_test[i]}')

#Előrejelzés a változóként beállított intervallumon
last_n_rows = np.array(df.drop(['Prediction'], axis=1))[-forecasting_days:]
forecasts = rbf.predict(last_n_rows)
print(forecasts)
print(df.tail(forecasting_days))