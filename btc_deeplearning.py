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
    array = np.reshape(array, (lenght, 1)) #Itt van a bug
    scaler = MinMaxScaler()
    scaled_set = scaler.fit_transform(array)
    ret_arr = [scaled_set[i][0] for i in range(0, len(scaled_set))] #nem szép megoldás, típuskonverzió miatt
    return ret_arr

#30 bementhez 1 kimenet rendelése
def processData(trainset, pastdays):
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
        y_train.append(temp_ytrain)
    return(x_train, y_train)

normalize(df)
training_set, test_set = split()
training_set = MinMaxScale(training_set)
#test_set = MinMaxScale(test_set)
pastdays = 30
X_train, Y_train = processData(training_set, pastdays)

print(X_train)
print("------------------------------------------------------------------------------------------------")
print(Y_train)