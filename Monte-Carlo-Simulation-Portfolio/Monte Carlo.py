import requests
import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import json
from scipy.stats import norm
import os
#See Styles
#print(plt.style.available)
        
class monte_carlo:
    def __init__(self, portfolioName, symbols, weights, num_simulations, predicted_days, upper_bound, lower_bound):
        self.portfolioName = portfolioName
        self.symbols = symbols
        self.weights = weights
        self.num_simulations = num_simulations
        self.predicted_days = predicted_days
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.path = os.path.join(os.path.expanduser("~"),'Saved Portfolios')

    def get_portfolio(self):            
        # Create Directory
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            
        # Get Symbols and Request Necessary Data
        base_string = ""

        for x in self.symbols:
            base_string = base_string + x + ','

        url = "https://api.iextrading.com/1.0/stock/market/batch?symbols=" + base_string + "&types=chart&range=5y&filter=date,close,changePercent"
        r = requests.get(url)
        j = r.text
        d = json.loads(j)

        dfs = []

        for symbol in self.symbols:
            data = d[symbol]['chart']

            df = pd.DataFrame.from_dict(data)
            df.columns = [symbol + " Percent Change", symbol + " Close", "Date"]
            df.set_index('Date', inplace=True)
            dfs.append(df)

        appended_data = pd.concat(dfs, axis=1)

        cols = [c for c in appended_data.columns if c.split(" ", 1)[1] not in 'Percent Change']

        port_val = appended_data[cols] * self.weights

        # Get position value
        appended_data['Portfolio Value'] = port_val.sum(axis=1)

        # Get columns that contain close
        cols = [col for col in appended_data.columns if 'Close' in col]
        dftemp = appended_data[cols]
        # Replace String 'Close'
        dftemp = dftemp.rename(columns={col: col.split(' ')[0] for col in dftemp.columns})

        for column, x in zip(dftemp, self.weights):
            colname = column
            appended_data[colname + ' Value'] = dftemp[column] * x
            appended_data[colname + ' Weight'] = appended_data[colname + ' Value'] / appended_data['Portfolio Value']

        # Calculate Portfolio Returns
        appended_data['Portfolio %'] = appended_data['Portfolio Value'].pct_change()
        appended_data = appended_data.dropna(how='any')
        appended_data = appended_data.reindex_axis(sorted(appended_data.columns), axis=1)

        appended_data.to_csv(self.path + self.portfolioName + ' Price Data.csv')

        self.prices = pd.Series(appended_data['Portfolio Value'])
        self.returns = pd.Series(appended_data['Portfolio %'])
        
    def load_portfolio(self):
        df = pd.read_csv(self.path + self.portfolioName + ' Price Data.csv', index_col=0)
        self.prices = pd.Series(df['Portfolio Value'])
        self.returns = pd.Series(df['Portfolio %'])
        
    def monte_carlo_sim(self):        
        returns = self.returns
        prices = self.prices
        
        last_price = prices[-1]
        
        simulation_df = pd.DataFrame()
 
        #Create Each Simulation as a Column in df
        for x in range(self.num_simulations):
            count = 0
            daily_vol = returns.std()
            
            price_series = []
            
            #Append Start Value
            price = last_price * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            
            #Series for Preditcted Days
            for i in range(self.predicted_days):
                if count == 251:
                    break
                price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
                count += 1
        
            simulation_df[x] = price_series
            self.simulation_df = simulation_df
            self.predicted_days = predicted_days

    def brownian_motion(self):
        returns = self.returns
        prices = self.prices
 
        last_price = prices[-1]
 
        #Note we are assuming drift here
        simulation_df = pd.DataFrame()
        
        #Create Each Simulation as a Column in df
        for x in range(self.num_simulations):
            
            #Inputs
            count = 0
            avg_daily_ret = returns.mean()
            variance = returns.var()
            
            daily_vol = returns.std()
            daily_drift = avg_daily_ret - (variance/2)
            drift = daily_drift - 0.5 * daily_vol ** 2
            
            #Append Start Value    
            prices = []
            
            shock = drift + daily_vol * np.random.normal()
            last_price * math.exp(shock)
            prices.append(last_price)
            
            for i in range(self.predicted_days):
                if count == 251:
                    break
                shock = drift + daily_vol * np.random.normal()
                price = prices[count] * math.exp(shock)
                prices.append(price)
                
        
                count += 1
            simulation_df[x] = prices
            self.simulation_df = simulation_df
            self.predicted_days = predicted_days

    def line_graph(self):
        prices = self.prices
        predicted_days = self.predicted_days
        simulation_df = self.simulation_df
        
        last_price = prices[-1]
        fig = plt.figure()
        style.use('bmh')
        
        title = "Monte Carlo Simulation: " + str(predicted_days) + " Days"
        plt.plot(simulation_df)
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Price ($USD)')
        plt.grid(True,color='grey')
        plt.axhline(y=last_price, color='r', linestyle='-')
        plt.show()

    def histogram(self):
        simulation_df = self.simulation_df
        
        ser = simulation_df.iloc[-1, :]
        x = ser
        mu = ser.mean()
        sigma = ser.std()
        
        num_bins = 20
        # the histogram of the data
        n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
         
        # add a 'best fit' line
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')
        plt.xlabel('Price')
        plt.ylabel('Probability')
        plt.title(r'Histogram of Speculated Stock Prices', fontsize=18, fontweight='bold')
 
        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.show()

    def VaR(self):
        simulation_df = self.simulation_df
        prices = self.prices
 
        last_price = prices[-1]
        
        price_array = simulation_df.iloc[-1, :]
        price_array = sorted(price_array, key=int)  
        var =  np.percentile(price_array, 1)
        
        val_at_risk = last_price - var
        print("Value at Risk: ", val_at_risk)
        
        #Histogram
        fit = norm.pdf(price_array, np.mean(price_array), np.std(price_array))
        plt.plot(price_array,fit,'-o')
        plt.hist(price_array,normed=True)
        plt.xlabel('Price')
        plt.ylabel('Probability')
        plt.title(r'Histogram of Speculated Stock Prices', fontsize=18, fontweight='bold')
        plt.axvline(x=var, color='r', linestyle='--', label='Price at Confidence Interval: ' + str(round(var, 2)))
        plt.axvline(x=last_price, color='k', linestyle='--', label = 'Current Stock Price: ' + str(round(last_price, 2)))
        plt.legend(loc="upper right")
        plt.show()

    def key_stats(self):
        simulation_df = self.simulation_df
 
        print('#------------------Simulation Stats------------------#')
        count = 1
        for column in simulation_df:
            print("Simulation", count, "Mean Price: ", simulation_df[column].mean())
            print("Simulation", count, "Median Price: ", simulation_df[column].median()) 
            count += 1
        
        print('\n')
        
        print('#----------------------Last Price Stats--------------------#')
        print("Mean Price: ", np.mean(simulation_df.iloc[-1,:]))
        print("Maximum Price: ",np.max(simulation_df.iloc[-1,:]))
        print("Minimum Price: ", np.min(simulation_df.iloc[-1,:]))
        print("Standard Deviation: ",np.std(simulation_df.iloc[-1,:]))
 
        print('\n')
       
        print('#----------------------Descriptive Stats-------------------#')
        price_array = simulation_df.iloc[-1, :]
        print(price_array.describe())
 
        print('\n')
               
        print('#--------------Annual Expected Returns for Trials-----------#')
        count = 1
        future_returns = simulation_df.pct_change()
        for column in future_returns:
            print("Simulation", count, "Annual Expected Return", "{0:.2f}%".format((future_returns[column].mean() * 252) * 100))
            print("Simulation", count, "Total Return", "{0:.2f}%".format((future_returns[column].iloc[1] / future_returns[column].iloc[-1] - 1) * 100))
            count += 1     
 
        print('\n')
                         
        #Create Column For Average Daily Price Across All Trials
        simulation_df['Average'] = simulation_df.mean(axis=1)
        ser = simulation_df['Average']
        
        print('#----------------------Percentiles--------------------------------#')
        percentile_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        for per in percentile_list:
            print("{}th Percentile: ".format(per), np.percentile(price_array, per))
        
        print('\n')
        
        print('#-----------------Calculate Probabilities-------------------------#')

        print(price_array)

        print("Probability Value is > "+str(lower_bound)+": ", "{0:.2f}%".format((sum(i > lower_bound for i in price_array) / len(price_array))* 100))
        print("Probability Value is < "+str(upper_bound)+": ", "{0:.2f}%".format((sum(i < upper_bound for i in price_array) / len(price_array))* 100))
        print("Probability Value is between > "+str(lower_bound)+" < "+str(upper_bound)+": ", "{0:.2f}%".format((sum(lower_bound <= i <= upper_bound for i in price_array) / len(price_array))* 100))

if __name__== "__main__":
    portfolioName = 'MC Port'
    symbols = ['AAPL', 'KO', 'HD', 'PM', 'RTN']
    weights = [1000,1000,2000,3000, 1000]
    num_simulations = 10
    predicted_days = 2000
    upper_bound = 1500000
    lower_bound = 998037
    sim = monte_carlo(portfolioName, symbols, weights, num_simulations, predicted_days, upper_bound, lower_bound)
    sim.get_portfolio()
    #sim.load_portfolio()
    sim.monte_carlo_sim()
    #sim.VaR()
    #sim.histogram()
    #sim.brownian_motion()
    sim.line_graph()
    sim.key_stats()
 
