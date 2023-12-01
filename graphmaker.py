import pandas as pd
import matplotlib.pyplot as plt

from lstm_gru import cryptos

for c in cryptos:
    f_data = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact_1201.csv', index_col='Date', parse_dates=['Date'])
    #fact_prices = f_data['2023-10-13':]['Close'].values

    p_data = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_olstm_wfdl.csv', index_col='Date', parse_dates=['Date'])
    #predicted_data = p_data['2023-10-13':]['Close'].values

    plt.clf()
    plt.plot(f_data['2023-10-13':]['Close'].values, color='black', label='Valós napi záróárfolyam')
    plt.plot(p_data['2023-10-13':]['Close'].values, color='green', label='Modell által előrejelzett árfolyam')
    plt.title(f'LSTM modell árfolyam előrejelzés validációja {c} kriptovalután')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel('Jósolt napok száma')
    plt.ylabel('Ár ($)')
    plt.legend(loc='upper left')
    plt.savefig(f'.\SVG Format Figures\{c}-USD_predict_valid_olstm_wfdl.svg', format='svg')
    plt.show()