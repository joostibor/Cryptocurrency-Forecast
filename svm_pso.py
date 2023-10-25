import pandas as pd
import numpy as np
import operator
import random

from lstm_gru import refreshExchangeRate, cryptos, visualizeAndSave
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from datetime import datetime, timedelta
from random import randint
from scipy.stats import uniform 

'''
def fitness(data, _C, _gamma):
    kernel = SVR(kernel='rbf', C=_C, gamma=_gamma)
    data['Prediction'] = data[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
    x = np.array(data.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
    x = x[:len(data)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
    y = np.array(data['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
    y = y[:-forecasting_days] #Üres sorok kihagyása
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő
    kernel.fit(x_train, y_train) #SVM tanítása
    #Pontosság vizsgálat
    model_accuracy = kernel.score(x_test, y_test)
    print(f"Modell pontossága: ", model_accuracy)
    return model_accuracy

def pso(data):
    n_particles = 30
    dimensions = 2
    w = 0.5
    c1 = 1
    c2 = 2
    particles = np.zeros((n_particles, 2))
    best_positions = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))
    accuracies = np.zeros((n_particles, 1))
    best_swarm_positions = [0,0]
    best_acc = -10.0
    for i in range(n_particles):
        particles[i] = np.random.uniform(0, 10, (1, 2))
        best_positions[i] = particles[i]
        accuracy = fitness(data, best_positions[i][0], best_positions[i][1])
        accuracies[i] = accuracy
        if accuracy > best_acc:
            best_swarm_positions[0] = particles[i][0]
            best_swarm_positions[1] = particles[i][1]
            best_acc = accuracy
        velocities[i] = np.random.uniform(0, 1000, (1, 2))

    while best_acc <= 0.6:
        for i in range(n_particles):
            for d in range(dimensions):
                r1 = np.random.uniform(0,1)
                r2 = np.random.uniform(0,1)
                cognitive_velocity = c1 * r1 * (best_positions[i][d] - particles[i][d])
                social_velocity = c2 * r2 * (best_swarm_positions[d] - particles[i][d]) 
                velocities[i][d] = w * velocities[i][d] + cognitive_velocity + social_velocity
            particles[i][0] = particles[i][0] + velocities[i][0]
            particles[i][1] = particles[i][1] + velocities[i][1]
            new_pos_acc = fitness(data, particles[i][0], particles[i][1])
            if new_pos_acc > accuracies[i]:
                best_positions[i] = particles[i]
                accuracies[i] = new_pos_acc
                if accuracies[i] > best_acc:
                    best_swarm_positions[0] = particles[i][0]
                    best_swarm_positions[1] = particles[i][1]
                    best_acc = accuracies[i]

        #print(best_swarm_positions)
        print(best_acc)

forecasting_days = 30
df = pd.read_csv(f'.\Exchange Rates\BTC-USD_p_osvm.csv', index_col=False)
pso(df)

'''

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
                  'gamma': uniform(0.0001, 10)}
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
    print(f"Modell pontossága {c}: ", model_accuracy)
    #Teszthalmazon átlagos négyzetes eltérés mérése
    predictions = svr.predict(x_test)
    rmse = (np.sqrt(np.mean(np.square((y_test - predictions) / y_test)))) * 100
    print(f"Átlagos négyzetes eltérés(%) {c}: ", rmse)
    #DataFrame visszaadása
    return dataframe

forecasting_days = 30 #Előrejelzendő napok száma
#refreshExchangeRate(cryptos) #Árfolyamadatok frissítése

for c in cryptos:
    df = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact.csv', index_col=False) #Adatfájl betöltése
    best_params = randomSearch(df)
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    rbf = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
    df= dataPreprocessingAndSvmTeaching(df, rbf)

    firsttime = True #.csv fájl import miatt létrehozott változó

    #Előrejelzés 30 napra
    for i in range(0, 30):
        if not firsttime:
            df = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_svm_random.csv', index_col=False)
            df = dataPreprocessingAndSvmTeaching(df, rbf)
        last_rows = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1))[-forecasting_days:] 
        forecast = rbf.predict(last_rows)
        df = df.drop(['Prediction'], axis=1)
        df.loc[len(df)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d %H:%M:%S%z') + timedelta(days=1))), forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], 0]
        df.to_csv(f'.\Exchange Rates\{c}-USD_p_svm_random.csv', index=False)
        firsttime = False

visualizeAndSave(cryptos, ['', '-USD_p_svm_random.csv'], ['', '-USD_svm_random.svg'])