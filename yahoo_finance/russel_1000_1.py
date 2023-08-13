import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# Define the Russell 1000 ETF symbol
russell_symbol = "IWB"

# Fetch Russell 1000 ETF data
russell_data = yf.Ticker(russell_symbol)
russell_history = russell_data.history(period="2y")

# Calculate RSI
russell_history['RSI'] = ta.momentum.RSIIndicator(russell_history['Close']).rsi()

# Calculate MACD
macd = ta.trend.MACD(russell_history['Close'])
russell_history['MACD'] = macd.macd()
russell_history['Signal_Line'] = macd.macd_signal()

# Calculate Bollinger Bands
bollinger = ta.volatility.BollingerBands(russell_history['Close'])
russell_history['Bollinger_High'] = bollinger.bollinger_hband()
russell_history['Bollinger_Low'] = bollinger.bollinger_lband()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(russell_history.index, russell_history['Close'], label='Russell 1000 ETF')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Russell 1000 ETF Price')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(russell_history.index, russell_history['RSI'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--', label='Overbought')
plt.axhline(y=30, color='g', linestyle='--', label='Oversold')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.title('Relative Strength Index (RSI)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(russell_history.index, russell_history['Close'], label='Russell 1000 ETF')
plt.plot(russell_history.index, russell_history['Bollinger_High'], label='Upper Bollinger Band', color='g')
plt.plot(russell_history.index, russell_history['Bollinger_Low'], label='Lower Bollinger Band', color='r')
plt.fill_between(russell_history.index, russell_history['Bollinger_High'], russell_history['Bollinger_Low'], alpha=0.2, color='gray')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bollinger Bands')
plt.legend()

plt.tight_layout()
plt.show()
