import pandas as pd
import numpy as np

from lstm_gru import refreshExchangeRate, cryptos, visualizeAndSave
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from datetime import datetime, timedelta
from scipy.stats import uniform 
from skopt import BayesSearchCV

def bayesSearch(dataframe):
    dataframe['Prediction'] = dataframe[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
    x = np.array(dataframe.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
    x = x[:len(dataframe)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
    y = np.array(dataframe['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
    y = y[:-forecasting_days] #Üres sorok kihagyása
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő
    bayes_search = BayesSearchCV(SVR(kernel='rbf'),
                                search_spaces=[{'C': (1e-6, 1e+3),
                                                'gamma': (1e-6, 1e+1)} ],
                                n_iter=50,
                                random_state=123)
    bayes_search.fit(x_train, y_train)
    return bayes_search.best_params_

def gridSearch(dataframe):
    dataframe['Prediction'] = dataframe[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
    x = np.array(dataframe.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
    x = x[:len(dataframe)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
    y = np.array(dataframe['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
    y = y[:-forecasting_days] #Üres sorok kihagyása
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő
    param_grid = {'C': [0.1, 1, 10, 50, 100, 500, 1000],
                  'gamma': [0.00001, 0.0001, 0.001, 0.001, 0.1, 1, 10]}
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid)
    grid_search.fit(x_train, y_train)
    return grid_search.best_params_

def randomSearch(dataframe):
    dataframe['Prediction'] = dataframe[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
    x = np.array(dataframe.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
    x = x[:len(dataframe)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
    y = np.array(dataframe['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
    y = y[:-forecasting_days] #Üres sorok kihagyása
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő
    param_dist = {'C': uniform(0, 1000),
                  'gamma': uniform(0.00001, 10)}
    svr = SVR()
    random_search = RandomizedSearchCV(estimator=svr, param_distributions=param_dist, cv=5, n_iter=50)
    random_search.fit(x_train, y_train)
    return random_search.best_params_

def dataPreprocessingAndSvmTeaching(dataframe, svr):
    dataframe['Prediction'] = dataframe[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
    x = np.array(dataframe.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
    x = x[:len(dataframe)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
    y = np.array(dataframe['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
    y = y[:-forecasting_days] #Üres sorok kihagyása
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő
    svr.fit(x_train, y_train) #SVM tanítása
    #Pontosság vizsgálat
    model_accuracy = svr.score(x_test, y_test)
    statfile = open("SVM_stats.txt", "a")
    statfile.write(f"Modell pontossaga {c}: {model_accuracy}\n")
    print(f"Modell pontossága {c}: ", model_accuracy)
    #Teszthalmazon átlagos négyzetes eltérés mérése
    predictions = svr.predict(x_test)
    rmse = (np.sqrt(np.mean(np.square((y_test - predictions) / y_test)))) * 100
    statfile.write(f"Atlagos negyzetes elteres(%) {c}: {rmse}\n")
    print(f"Atlagos negyzetes elteres(%) {c}: ", rmse)
    statfile.close()
    #DataFrame visszaadása
    return dataframe

forecasting_days = 30 #Előrejelzendő napok száma
#refreshExchangeRate(cryptos) #Árfolyamadatok frissítése

for c in cryptos:
    statfile = open("SVM_stats.txt", "a")
    statfile.write(f"\n----------------------SVM_w_bayes_{c}----------------------\n")
    statfile.close()
    df = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact.csv', index_col=False) #Adatfájl betöltése
    best_params = bayesSearch(df)
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    statfile = open("SVM_stats.txt", "a")
    statfile.write(f"Legjobb C: {best_C}, legjobb gamma: {best_gamma}\n")
    statfile.close()
    rbf = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
    df= dataPreprocessingAndSvmTeaching(df, rbf)

    firsttime = True #.csv fájl import miatt létrehozott változó

    #Előrejelzés 30 napra
    for i in range(0, 30):
        if not firsttime:
            df = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_svm_bayes_2.csv', index_col=False)
            df = dataPreprocessingAndSvmTeaching(df, rbf)
        last_rows = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1))[-forecasting_days:] 
        forecast = rbf.predict(last_rows)
        df = df.drop(['Prediction'], axis=1)
        df.loc[len(df)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d %H:%M:%S%z') + timedelta(days=1))), forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], 0]
        df.to_csv(f'.\Exchange Rates\{c}-USD_p_svm_bayes_2.csv', index=False)
        firsttime = False

visualizeAndSave(cryptos, ['', '-USD_p_svm_bayes_2.csv'], ['', '-USD_svm_bayes_2.svg'])