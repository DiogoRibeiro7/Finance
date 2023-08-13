import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the Russell 1000 ETF symbol
russell_symbol = "IWB"

# Fetch Russell 1000 ETF data
russell_data = yf.Ticker(russell_symbol)
russell_history = russell_data.history(period="1y")

# Calculate daily returns
russell_history['Daily_Return'] = russell_history['Close'].pct_change()

# Calculate rolling volatility (standard deviation of returns)
rolling_window = 30  # You can adjust the window size
russell_history['Volatility'] = russell_history['Daily_Return'].rolling(
    window=rolling_window).std()

# Calculate rolling average return
russell_history['Rolling_Avg_Return'] = russell_history['Daily_Return'].rolling(
    window=rolling_window).mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(russell_history.index,
         russell_history['Rolling_Avg_Return'], label='Rolling Avg Return')
plt.plot(russell_history.index,
         russell_history['Volatility'], label='Volatility')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Russell 1000 ETF Analysis')
plt.legend()
plt.grid(True)
plt.show()



# Calculate daily returns
russell_history['Daily_Return'] = russell_history['Close'].pct_change()

# Calculate 50-day and 200-day moving averages
russell_history['50_MA'] = russell_history['Close'].rolling(window=50).mean()
russell_history['200_MA'] = russell_history['Close'].rolling(window=200).mean()

# Calculate correlation with S&P 500 index
sp500_data = yf.Ticker("^GSPC").history(period="2y")
correlation = russell_history['Daily_Return'].corr(sp500_data['Close'])

# Calculate annualized volatility
annualized_volatility = russell_history['Daily_Return'].std() * np.sqrt(252)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(russell_history.index,
         russell_history['Close'], label='Russell 1000 ETF')
plt.plot(russell_history.index, russell_history['50_MA'], label='50-day MA')
plt.plot(russell_history.index, russell_history['200_MA'], label='200-day MA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Russell 1000 ETF Price and Moving Averages')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(russell_history.index,
         russell_history['Daily_Return'], label='Daily Returns')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('Daily Returns and Volatility')
plt.legend()

plt.tight_layout()
plt.show()

print("Correlation with S&P 500:", correlation)
print("Annualized Volatility:", annualized_volatility)



# Calculate 20-day moving average
russell_history['20_MA'] = russell_history['Close'].rolling(window=20).mean()

# Calculate 20-day standard deviation
russell_history['20_STD'] = russell_history['Close'].rolling(window=20).std()

# Calculate upper and lower Bollinger Bands
russell_history['Upper_Band'] = russell_history['20_MA'] + 2 * russell_history['20_STD']
russell_history['Lower_Band'] = russell_history['20_MA'] - 2 * russell_history['20_STD']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(russell_history.index, russell_history['Close'], label='Russell 1000 ETF')
plt.plot(russell_history.index, russell_history['20_MA'], label='20-day MA', linestyle='--')
plt.plot(russell_history.index, russell_history['Upper_Band'], label='Upper Bollinger Band', color='g')
plt.plot(russell_history.index, russell_history['Lower_Band'], label='Lower Bollinger Band', color='r')
plt.fill_between(russell_history.index, russell_history['Upper_Band'], russell_history['Lower_Band'], alpha=0.2, color='gray')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Russell 1000 ETF Bollinger Bands')
plt.legend()
plt.grid(True)
plt.show()
