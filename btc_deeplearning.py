import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('BTC-USD.csv')

def normalize(dataframe):
    #Felesleges oszlopok törlése
    del dataframe['Adj Close']
    del dataframe['Date']
    
    #Átlagár oszlop hozzáadása
    avg_price = [(row['Open'] + row['Close'] + row['High'] + row['Low'])/4 for index, row in df.iterrows()]
    dataframe['Avg'] = avg_price

    #Min-Max normlizáció
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(dataframe)
    normalized_df = pd.DataFrame(normalized_data, columns=dataframe.columns)

    return normalized_df

df = normalize(df)
print(df)