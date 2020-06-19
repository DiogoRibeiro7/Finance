import numpy as np

#store the variables in arrays
prob = np.array([0.25, 0.5, 0.25])
rate_1 = np.array([0.05, 0.075, 0.10])
rate_2 = np.array([0.2, 0.15, 0.1])

# expected return of each investment
expected_return1 = np.sum(prob * rate_1)
expected_return2 = np.sum(prob * rate_2)

# expected return of the equally weighted portfolio
weights = np.array([0.5, 0.5])
individual_returns = np.array([rate_1, rate_2])
portfolio_returns = np.dot(weights, individual_returns)


# covariance matrix given probabilities
cov_matrix = np.cov(rate_1, rate_2, ddof=0, aweights=prob)

print(cov_matrix)


#  variance and standard deviation of each investment
var1 = cov_matrix[0,0]  # variance of any asset is the covariance of its returns WITH its returns
var2 = cov_matrix[1,1]  # variance of any asset is the covariance of its returns WITH its returns
std1 = np.sqrt(var1)  # std deviation is simply the square root of the variance
std2 = np.sqrt(var2)  # std deviation is simply the square root of the variance

#  correlation between Asset 1 & 2's returns
cov = cov_matrix[0,1]
corr = cov / (std1 * std2)  # correlation of returns between 2 assets = covariance of their returns / (their std multiplied)

#  variance of portfolio
portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))

# standard deviation (volatility of the portfolio)
portfolio_vols = np.sqrt(portfolio_var)


#  just a function that returns a percentile for a given float
def percentage (number):
    return str(np.round(number, 4) * 100) + '%'

#  print the various variables for intepretation 
print('Expected Return of Investment 1 = {}'.format(percentage(expected_return1)))
print('Expected Return of Investment 2 = {}'.format(percentage(expected_return2)))
print('Expected Return of Portfolio = {}'.format(percentage(portfolio_returns)))
print('Standard Deviation of Investment 1 = {}'.format(percentage(std1)))
print('Standard Deviation of Investment 1 = {}'.format(percentage(std2)))
print('Correlation between Returns of 1 & 2 = {}'.format(round(corr, 4)))
print('Risk of Portfilio = {}'.format(percentage(portfolio_vols)))