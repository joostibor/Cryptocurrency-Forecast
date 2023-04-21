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
    training = []
    test = []
    div_ratio = 0.8
    df_elemenets_count = len(df.index)
    div_at = int(df_elemenets_count * div_ratio)
    for index, row in df.iterrows():
        if index < div_at:
            training.append(df.iloc[index].at['Avg'])
        else:
            test.append(df.iloc[index].at['Avg'])
    return(training, test)

#MinMax normálizálás
def MinMaxScale(array):
    lenght = len(array)
    arr = np.array(array).reshape(lenght, 1)
    scaler = MinMaxScaler()
    scaled_set = scaler.fit_transform(arr)
    return scaled_set

def split2(dataframe, pastdays):
    training = []
    test = []
    for index, row in df.iterrows():
        outindex = index + pastdays
        if outindex > df.shape[0] - 1:
            break
        temp_training = dataframe.iloc[index:outindex].at['Avg']
        temp_test = dataframe.iloc[outindex].at['Avg']
        training.append(temp_training)

normalize(df)
training_set, test_set = split()
training_set = MinMaxScale(training_set)
test_set = MinMaxScale(test_set)

pastdays = 30
print(training_set)
print("------------------------------------------------------------------------------------------------")
print(test_set)