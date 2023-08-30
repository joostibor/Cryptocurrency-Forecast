import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

p_data = pd.read_csv('BTC-USD_predict_wfdl.csv', index_col='Date', parse_dates=['Date'])
f_data = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=['Date'])

predict_data = p_data['2021':]
fact_data = f_data['2021':]

plt.plot(p_data['2021':]['Close'], color='green', label='Modell alapján előrejelzett napi záró árfolyam')
plt.plot(fact_data['2021':]['Close'], color='black', label='Valós napi záró árfolyamok')
plt.title('BTC záró árfolyam előrejelzés LSTM-GRU hibrid modellel')
plt.xticks(rotation=45, fontsize=8)
plt.ylabel('Ár ($)')
plt.legend(loc='upper left')
plt.savefig('test.svg', format='svg')
plt.show()