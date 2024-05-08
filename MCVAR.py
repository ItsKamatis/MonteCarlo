import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import pandas_datareader as pdr

#Get Data
def get_data(stock, start, end):
    stock_data = yf.download(stock, start, end)['Adj Close']
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

stocklist = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
stocks = [stock + '' for stock in stocklist]
end = dt.datetime.today()
start = end - dt.timedelta(days=300)

mean_returns, cov_matrix = get_data(stocks, start, end)

weights = np.random.random(len(mean_returns))
weights /= np.sum(weights)

#Monte Carlo Simulation
#Number of iterations
mc_sims = 100
T = 100 #timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=mean_returns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initial_investment = 100000

for m in range (0, mc_sims):
    #MC loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(cov_matrix)
    daily_returns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights,
                                               daily_returns.T)+1)*initial_investment

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Portfolio')
plt.show()


