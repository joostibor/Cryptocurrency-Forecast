import yfinance as yf
import datetime as dt

cryptos = ['BTC', 'ETH', 'BNB', 'DOGE', 'LTC'] #Bitcoin, Ethereum, Binance, DogeCoin, Litecoin

for i in range (0,len(cryptos)):
    crypto_exc = yf.Ticker(f'{cryptos[i]}-USD')
    hist = crypto_exc.history(start="2009-01-03", end=dt.datetime.now()) #Kezdődátum megegyezik a Bitcoin indulási idejével
    hist.drop(hist.columns[[5, 6]], axis=1, inplace=True) #Felesleges oszlopok törlése
    hist.to_csv(f'.\Exchange Rate Datas\{cryptos[i]}-USD_fact.csv', sep=',',index=True)