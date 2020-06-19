# Import the Python's number crunchers

from pandas_datareader import data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Pull Adjusted closing prices for the portfolio (consisting of 5 stocks) we want to construct

assets =  ['AAPL', 'GM', 'GE', 'FB', 'WMT']

df = pd.DataFrame()

for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo',start='2015-1-1' ,end='2017-1-1')['Adj Close']
    
df.head()

# Check the daily returns

d_returns = df.pct_change()
d_returns.head()


# Construct a covariance matrix for the portfolio's daily returns with the .cov() method

cov_matrix_d = d_returns.cov()
cov_matrix_d

# Annualise the daily covariance matrix with the standard 250 trading days

cov_matrix_a = cov_matrix_d * 250
cov_matrix_a

# Assign equal weights to the five stocks. Weights must = 1 so 0.2 for each 

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
weights


# Calculate the variance with the formula

port_variance = np.dot(weights.T, np.dot(cov_matrix_a, weights))
port_variance

# Just converting the variance float into a percentage

print(str(round(port_variance, 4) * 100) + '%')

# The standard deviation of a portfolio is just a square root of its variance

port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_a, weights)))
port_volatility

d_returns.plot(figsize=(16,8))
plt.show()

# basic stats on daily returns

d_returns.describe()


# correlation matrix of daily returns

d_returns.corr()

# annual standard deviation of Apple stock 

d_returns['AAPL'].std() * np.sqrt(250)