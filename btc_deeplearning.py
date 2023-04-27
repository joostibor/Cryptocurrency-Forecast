import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU

#Adattábla beolvasása DataFramebe
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
    array = np.reshape(array, (lenght, 1))
    scaler = MinMaxScaler()
    scaled_set = scaler.fit_transform(array)
    ret_arr = [scaled_set[i][0] for i in range(0, len(scaled_set))] #nem szép megoldás, típuskonverzió miatt
    return ret_arr

#30 bementhez 1 kimenet rendelése
def processData(trainset, timestamps):
    x_train = []
    y_train = []
    n = len(trainset)
    for i in range (0, n):
        maxindex = i + timestamps
        if maxindex > n - 1:
            break
        temp_xtrain = trainset[i:maxindex]
        temp_ytrain = trainset[maxindex]
        x_train.append(temp_xtrain)
        y_train.append(temp_ytrain)
    return(x_train, y_train)

def processTestData(testset):
    x_test = []
    n = len(testset)
    for i in range(0, n):
        maxindex = i + timestamps
        if maxindex > n - 1:
            break
        temp_xtest = testset[i:maxindex]
        x_test.append(temp_xtest)
    return x_test

def trainAndSaveModel(x_train, y_train):
    #LSTM modell
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1],1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=1, activation="relu"))

    #GRU modell
    gru_model = Sequential()
    gru_model.add(GRU(units=30, return_sequences=True, activation='tanh'))
    gru_model.add(Dropout(0.2))
    gru_model.add(Dense(units=1, activation="relu"))

    #Összevont modell
    merged_model = Sequential()
    merged_model.add(lstm_model)
    merged_model.add(gru_model)
    merged_model.add(Dense(units=1, activation="relu"))
    merged_model.compile(optimizer='adam',loss='mean_squared_error')
    merged_model.fit(x_train, y_train, epochs=100)

    merged_model.save('btc_model.h5')

normalize(df)
training_set, test_set = split()
training_set = MinMaxScale(training_set)

#Tanító tömbök kialakítása
timestamps = 30
X_train, Y_train = processData(training_set, timestamps)
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#Tanítás és Mentés
trainAndSaveModel(X_train, Y_train)

#Teszt set előkészítése és előrejelzés
model = load_model('btc_model.h5')
lenght = len(test_set)
test_set = np.reshape(test_set, (lenght, 1))
scaler = MinMaxScaler()
scaled_test = scaler.fit_transform(test_set)
X_test = processTestData(scaled_test)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predict = model.predict(X_test)
predicted_price = np.max(predict[0])
predicted_price = np.reshape(predicted_price, (1,-1))
predicted_price = scaler.inverse_transform(predicted_price)
print(predicted_price)