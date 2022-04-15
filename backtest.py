
from turtle import pd
import numpy as np
import pandas as pd
from tech_ind import *
from OracleStrategy import *
from TechnicalStrategy import *

def assess_strategy(data = None, starting_value = 1000000, fixed_cost = 9.95, floating_cost = 0.005):
    #create dataframe for trades
    trades = data
    start_date = trades.index[0]
    end_date = trades.index[-1]

    dates = pd.date_range(start = start_date, end = end_date)
    symbl_list = trades.loc[:,"Symbol"].drop_duplicates().to_list()   

    #create DataFrame that holds the market data for each stock we buy or sell
    
    new_df = pd.DataFrame(index = dates)
    market = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True, usecols= ["Date", "Adj Close"])
    market = new_df.join(market)
    market.rename(columns = {"Adj Close":'CASH'}, inplace = True)
    market= market.dropna()
    market["CASH"] = starting_value
    for i in symbl_list: 
        column = pd.read_csv("data/" + i + ".csv", index_col="Date", parse_dates=True, usecols= ["Date", "Adj Close"])
        column.rename(columns = {"Adj Close":i}, inplace = True)
        market = market.join(column, how = 'inner')


    #create dataframe that contains portfolio allocations
    portfolio = market.copy()
    market["CASH"] = 1

    #fill dataframe with trades that we make 

    portfolio.iloc[:, 1:] = 0 

    for j in range(0, trades.shape[0]): 
        start = trades.index[j]
        symbol = trades.iloc[j, 0]
        trans_val = trades.iloc[j, 2]*market.loc[start,symbol]
        if trades.iloc[j, 1] == "BUY":
            cash = portfolio.loc[start, "CASH"] - trans_val - (fixed_cost + floating_cost*trans_val)
            portfolio.loc[start:end_date, symbol] = portfolio.loc[start:end_date, symbol] + trades.iloc[j, 2]
            portfolio.loc[start:end_date, "CASH"] = cash

        
        elif trades.iloc[j, 1] == "SELL":
            cash = portfolio.loc[start:, "CASH"] + trans_val - (fixed_cost + floating_cost*trans_val)
            portfolio.loc[start:end_date, symbol] = portfolio.loc[start:end_date, symbol] - trades.iloc[j, 2]
            portfolio.loc[start:end_date, "CASH"] = cash

        else:

    print("here")
    print(portfolio)

    #create daily_portforlio_values

    daily_portforlio_values = pd.DataFrame(index = portfolio.index)
    daily_portforlio_values["Portfolio"] = pd.DataFrame(portfolio*market).sum(axis = 1)
    return(daily_portforlio_values)
    


print(assess_strategy(data = OracleStrategy.test()))