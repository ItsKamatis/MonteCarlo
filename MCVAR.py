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

stocklist = ['NVDA', 'NOC', 'AAL', 'FSHOX', 'TSLA']
stocks = [stock + '' for stock in stocklist]
end = dt.datetime.today()
start = end - dt.timedelta(days=300)

mean_returns, cov_matrix = get_data(stocks, start, end)

weights = np.random.random(len(mean_returns))
weights /= np.sum(weights)
#############################
#Monte Carlo Simulation
#Number of iterations
mc_sims = 1000
T = 120 #timeframe in days
#############################
meanM = np.full(shape=(T, len(weights)), fill_value=mean_returns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
########################
initial_investment = 300000
########################
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

def mcvar(returns, alpha):
    #Input series of returns
    #out % return distribution to given confidence value of alpha
    if isinstance(returns, pd.Series):
        returns = np.percentile(returns, alpha)
    else:
        raise TypeError('Expected a pandas Series')
    return np.percentile(returns, alpha)


#Make sure to change the alpha value to 5% or 95% for the desired confidence level
alpha = 2
def mcCvar(returns, alpha):
    #Input series of returns
    #out Cvar or expected shortfall return distribution to given confidence
    # level of alpha
    if isinstance(returns, pd.Series):
        belowVar = returns <= np.percentile(returns, alpha)
        return returns[belowVar].mean()
    else:
        raise TypeError('Expected a pandas Series')

portfolio_returns = pd.Series(portfolio_sims[-1, :])

var = initial_investment - mcvar(portfolio_returns, alpha)
mcvar = initial_investment - mcCvar(portfolio_returns, alpha)

print('Value at Risk: ${}'.format(round(var, 2)))
print('Conditional Value at Risk: ${}'.format(round(mcvar, 2)))


