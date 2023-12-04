import pandas as pd
import matplotlib.pyplot as plt

from lstm_gru import cryptos

for c in cryptos:
    f_data = pd.read_csv(f'.\Exchange Rates\{c}-USD_fact_1111.csv', index_col='Date', parse_dates=['Date'])

    p_data = pd.read_csv(f'.\Exchange Rates\{c}-USD_p_svm_evolution.csv', index_col='Date', parse_dates=['Date'])

    plt.clf()
    plt.plot(f_data['2023-10-13':]['Close'].values, color='black', label='Valós napi záróárfolyam')
    plt.plot(p_data['2023-10-13':]['Close'].values, color='green', label='Modell által előrejelzett árfolyam')
    plt.title(f'Evolúciós algoritmussal optimalizált SVM modell\n validációja {c} kriptovalután')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel('Jósolt napok száma')
    plt.ylabel('Ár ($)')
    plt.legend(loc='upper left')
    plt.savefig(f'.\SVG Format Figures\{c}-USD_predict_valid_svm_evolution.svg', format='svg')
    plt.show()