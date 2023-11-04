import random
import pandas as pd
import numpy as np

from operator import itemgetter
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from lstm_gru import cryptos, visualizeAndSave
from datetime import datetime, timedelta

#Egyedek fitness értékének számítása
def fitness(individual):
    svr = SVR(kernel='rbf', C=individual[0], gamma=individual[1])
    svr.fit(x_train, y_train)
    fitn = svr.score(x_test, y_test)
    print(fitn)
    return fitn

#Összes egyed rátermettségének számítása
def sortPopulationByFitness(population):
    for i in range(len(population)):
        population[i][2] = fitness(population[i])
    return sorted(population, key=itemgetter(2), reverse=True)

#Következő generációhoz legjobb egyedek, illetve szerencsések hozzáadása
def selectForNextGen(sorted_popul, best_x, luckies):
    next_gen = []
    #Legjobbak hozzáadása
    for i in range(best_x):
        next_gen.append(sorted_popul[i])
    #Szerencse alapon bekerült egyedek hozzáadása
    for i in range(luckies):
        lucky_boy = random.choice(sorted_popul)
        next_gen.append(lucky_boy)
    random.shuffle(next_gen)
    return next_gen

#Gyermek egyed létrehozása
def makeChildIndividual(parent1, parent2):
    child = [parent1[0], parent2[1], 0]
    return child

#Következő generációhoz egyedek hozzáadása keresztezés által, illetve új egyedekkel való feltölés
def makeNextGen(population):
    next_gen = population
    #Keresztezés 
    for i in range(0, (len(population)-1), 2):
        next_gen.append(makeChildIndividual(population[i], population[i+2]))
        next_gen.append(makeChildIndividual(population[i+2], population[i]))
    next_gen_n = len(next_gen)
    for i in range(next_gen_n, individuals):
        next_gen.append([random.uniform(1e-1, 1e+3), random.uniform(1e-5, 1e+1), 0])
    return next_gen

#Véletlenszerű mutálás, mutálási esély 20%
def mutation(individual): 
    idx = random.randint(0,1)
    if idx==0:
        individual[idx] = random.uniform(1e-1, 1e+3)
    else:
        individual[idx] = random.uniform(1e-5, 1e+1)
    return individual

def mutatePopulation(population, mutation_chance):
    for i in range(len(population)):
        if random.random() * 100 < mutation_chance:
            population[i] = mutation(population[i])

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

population = []
individuals = 50
generations = 10
forecasting_days = 30

for c in cryptos:
    #Preprocessing
    df = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact.csv', index_col=False) #Adatfájl betöltése
    df['Prediction'] = df[['Close']].shift(-forecasting_days) #Új oszlop beszúrása az előrejelzendő napok számával megegyező eltolással
    x = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1)) #Close oszlop átkonvertálás numpy tömbbé
    x = x[:len(df)-forecasting_days] #x tömb méretének csökkentése az előjelzendő napok számával
    y = np.array(df['Prediction']) #Újonnan beszúrt oszlop konvertálása numpy tömbbé
    y = y[:-forecasting_days] #Üres sorok kihagyása
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #Adatok szétválasztása -> 80% tanító, 20% tesztelő

    #Evolúciós algoritmus
    #Első populáció egyedeinek generálása
    for i in range(individuals):
        population.append([random.uniform(1e-1, 1e+3), random.uniform(1e-5, 1e+1), 0])

    #Végigiterálás a generációkon
    for i in range(generations):
        print(f'{i}. generáció')
        sorted_population = sortPopulationByFitness(population)
        print(sorted_population)
        parents_population = selectForNextGen(sorted_population, 5, 5)
        population = makeNextGen(parents_population)
        mutatePopulation(population, 30)

    print(sorted_population[0][0], sorted_population[0][1])
    print(sorted_population[0][2])

    #Előrejelzés
    best_C = sorted_population[0][0]
    best_gamma = sorted_population[0][1]
    rbf = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
    rbf.fit(x_train, y_train)
    model_accuracy = rbf.score(x_test, y_test)
    print(f"Modell pontossága {c}: ", model_accuracy)
    predictions = rbf.predict(x_test)
    rmse = (np.sqrt(np.mean(np.square((y_test - predictions) / y_test)))) * 100 #teszthalmazon
    print(f"Átlagos négyzetes eltérés(%) {c}: ", rmse)

    firsttime = True #.csv fájl import miatt létrehozott változó

    #Előrejelzés 30 napra
    for i in range(0, 30):
        if not firsttime:
            df = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_svm_evolution.csv', index_col=False)
            df = dataPreprocessingAndSvmTeaching(df, rbf)
        last_rows = np.array(df.drop(['Date', 'Open', 'High', 'Low', 'Volume', 'Prediction'], axis=1))[-forecasting_days:] 
        forecast = rbf.predict(last_rows)
        df = df.drop(['Prediction'], axis=1)
        df.loc[len(df)] = [str((datetime.strptime(df.iloc[-1]['Date'], '%Y-%m-%d %H:%M:%S%z') + timedelta(days=1))), forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], forecast[forecasting_days-1], 0]
        df.to_csv(f'.\Exchange Rates\{c}-USD_p_svm_evolution.csv', index=False)
        firsttime = False

visualizeAndSave(cryptos, ['', '-USD_p_svm_evolution.csv'], ['', '-USD_svm_evolution.svg'])