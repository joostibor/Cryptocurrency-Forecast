import pandas as pd

df = pd.read_csv('BTC-USD.csv')

print(df.iloc[0].to_string())

def normalize (dataframe):
    del df['unix']
    del df['symbol']

