import matplotlib.pyplot as plt
import pandas as pd

p_data = pd.read_csv('BTC-USD_predict.csv', index_col='Date', parse_dates=['Date'])
f_data = pd.read_csv('BTC-USD_fact.csv', index_col='Date', parse_dates=['Date'])

predict_data = p_data['2022':]
predict_prices = predict_data['Close'].values
fact_data = f_data['2022':]
fact_prices = fact_data['Close'].values

plt.plot(predict_prices, color='green', label='Modell alapján előrejelzett napi záró árfolyam')
plt.plot(fact_prices, color='black', label='Valós napi záró árfolyamok')
plt.title('BTC záró árfolyam előrejelzés LSTM-GRU hibrid modellel')
plt.xlabel('Dátum')
plt.ylabel('Ár')
plt.legend(loc='upper left')
plt.show()