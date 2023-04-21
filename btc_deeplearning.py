import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('BTC-USD.csv')

#Bemeneti dataset alap normalizálása
def normalize(dataframe):
    #Felesleges oszlopok törlése
    del dataframe['Adj Close']
    del dataframe['Date']
    
    #Átlagár oszlop hozzáadása
    avg_price = [(row['Open'] + row['Close'] + row['High'] + row['Low'])/4 for index, row in dataframe.iterrows()]
    dataframe['Avg'] = avg_price

#Dataset szétosztása training és test setre a forrás pszeudokódja alapján
#A kiválasztott érték a tanításhoz az avg
def split():
    training = np.array([])
    test = np.array([])
    div_ratio = 0.8
    df_elemenets_count = len(df.index)
    div_at = int(df_elemenets_count * div_ratio)
    for index, row in df.iterrows():
        if index < div_at:
            training = np.append(training, df.iloc[index].at['Avg'])
        else:
            test = np.append(test, df.iloc[index].at['Avg'])
    return(training, test)

#MinMax normálizálás
def MinMaxScale(array):
    lenght = len(array)
    print(type(array[0]))
    array = np.reshape(array, (lenght, 1)) #Itt van a bug
    print(type(array[0]))
    scaler = MinMaxScaler()
    scaled_set = scaler.fit_transform(array)
    print(type(scaled_set[0]))
    return scaled_set

#30 bementhez 1 kimenet rendelése
def processData(trainset, pastdays):
    #print(type(trainset[0]))
    x_train = []
    y_train = []
    n = len(trainset)
    for i in range (0, n):
        maxindex = i + pastdays
        if maxindex > n - 1:
            break
        temp_xtrain = trainset[i:maxindex]
        temp_ytrain = trainset[maxindex]
        x_train.append(temp_xtrain)
        y_train.append(float(temp_ytrain))
    return(x_train, y_train)

normalize(df)
training_set, test_set = split()
print(type(training_set[0]))
training_set = MinMaxScale(training_set)
print(type(training_set[0]))
#test_set = MinMaxScale(test_set)
pastdays = 30
X_train, Y_train = processData(training_set, pastdays)

#print(X_train)
print("------------------------------------------------------------------------------------------------")
#print(Y_train)