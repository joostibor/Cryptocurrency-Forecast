import pandas as pd

df = pd.read_csv('BTC-USD.csv')

def normalize(dataframe):
    del dataframe['Adj Close']
    del dataframe['Date']
    avg_price = [(row['Open'] + row['Close'] + row['High'] + row['Low'])/4 for index, row in df.iterrows()]
    dataframe['Avg'] = avg_price

normalize(df)
print(df)