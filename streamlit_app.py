import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Imposta il titolo dell'applicazione
st.title("Bitcoin Price Analysis with Power Law Projection")

# Numero di anni per la proiezione
projection_length_in_years = 5

# Numero di giorni per la proiezione
projection_length_in_days = 365 * projection_length_in_years

# Scarica i dati di prezzo del Bitcoin
url = 'https://data.bitcoinity.org/export_data.csv?c=e&currency=USD&data_type=price&r=day&t=l&timespan=all'
bitcoin_data = pd.read_csv(url, parse_dates=['Time'])

# Prezzi iniziali dalla colonna 10 (indice 9), quindi passa alla colonna 4 (indice 3) alla riga 560
initial_prices = bitcoin_data.iloc[:560, 9]
subsequent_prices = bitcoin_data.iloc[560:, 3]
prices = pd.concat([initial_prices, subsequent_prices])

# Assicurati che la datetime sia timezone-naive
bitcoin_data['Time'] = bitcoin_data['Time'].dt.tz_localize(None)

# Combina i tempi e i prezzi in un unico DataFrame
combined_data = pd.DataFrame({
    'Time': bitcoin_data['Time'],
    'Price': prices
})

# Rimuovi le righe con valori NaN nella colonna 'Price'
combined_data = combined_data.dropna(subset=['Price'])

# Calcola i giorni dal Genesis Block (2009-01-03)
genesis_block = pd.Timestamp('2009-01-03')
combined_data['Days From Genesis'] = (combined_data['Time'] - genesis_block).dt.days

# Esegui la trasformazione log-log
log_days = np.log10(combined_data['Days From Genesis'])
log_price = np.log10(combined_data['Price'])

# Esegui la regressione lineare sui dati log-log
slope, intercept, r_value, p_value, std_err = linregress(log_days, log_price)

# Genera la linea di adattamento della legge di potenza
# Inizia prendendo il giorno di partenza e seleziona il giorno di arresto
start = combined_data['Days From Genesis'].min()
stop = combined_data['Days From Genesis'].max() + projection_length_in_days
x_fit = np.linspace(start, stop, (stop - start))

# Calcola i valori di y
y_fit = 10 ** (slope * np.log10(x_fit) + intercept)

# Creazione dei grafici con Matplotlib e visualizzazione in Streamlit
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Grafico in scala lineare
axes[0].plot(combined_data['Days From Genesis'], combined_data['Price'], label='Data')
axes[0].plot(x_fit, y_fit, 'r-', linewidth=2, label='Power Law Fit')
axes[0].set_xlabel('Days From Genesis Block')
axes[0].set_ylabel('Price (USD)')
axes[0].set_title(f'Bitcoin Price - Linear Scale\nSlope: {slope:.2f}, Intercept: {intercept:.2f}, R^2: {r_value**2:.2f}')
axes[0].legend()
axes[0].grid(True)

# Grafico in scala logaritmica (asse y)
axes[1].plot(combined_data['Days From Genesis'], combined_data['Price'], label='Data')
axes[1].plot(x_fit, y_fit, 'r-', linewidth=2, label='Power Law Fit')
axes[1].set_yscale('log')
axes[1].set_xlabel('Days From Genesis Block')
axes[1].set_ylabel('Price (USD)')
axes[1].set_title(f'Bitcoin Price - Log-Linear Scale\nSlope: {slope:.2f}, Intercept: {intercept:.2f}, R^2: {r_value**2:.2f}')
axes[1].legend()
axes[1].grid(True)

# Grafico in scala log-log
axes[2].plot(combined_data['Days From Genesis'], combined_data['Price'], label='Data')
axes[2].plot(x_fit, y_fit, 'r-', linewidth=2, label='Power Law Fit')
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_xlabel('Days From Genesis Block')
axes[2].set_ylabel('Price (USD)')
axes[2].set_title(f'Bitcoin Price - Log-Log Scale\nSlope: {slope:.2f}, Intercept: {intercept:.2f}, R^2: {r_value**2:.2f}')
axes[2].legend()
axes[2].grid(True)

# Imposta il layout del grafico
plt.tight_layout()

# Mostra il grafico in Streamlit
st.pyplot(fig)
