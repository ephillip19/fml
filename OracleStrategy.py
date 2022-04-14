# OracleStrategy.py
# CSCI 3465
from assess import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class OracleStrategy:
    def __init__(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def train(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def test(
        start_date="2018-01-01",
        end_date="2019-12-31",
        symbol="DIS",
        starting_cash=200000,
    ):
        # Inputs represent the date range to consider, the single stock to trade, and the starting portfolio value.
        #
        # Return a date-indexed DataFrame with a single column containing the desired trade for that date.
        # Given the position limits, the only possible values are -2000, -1000, 0, 1000, 2000.

        data = get_data(
            start_date,
            end_date,
            [symbol],
            column_name="Adj Close",
            include_spy=False,
        )

        df_trades = data.copy()
        df_trades.drop(columns=["DIS"])
        df_trades["Trade"] = np.NaN
        df_trades["Cash"] = 200000
        df_trades["Shares"] = 0

        for i in range(df_trades.shape[0]):
            today = df_trades.index[i]
            tomorrow = df_trades.index[i + 1]
            if data[today, "DIS"] < data[tomorrow, "DIS"]:
                if df_trades[today, "Shares"] == 0:
                    df_trades[today, "Trades"] = 1000
                    df_trades[tomorrow, "Shares"] = 1000
                    df_trades[tomorrow, "Cash"] = df_trades[tomorrow, "Shares"]*df_trades[tomorrow, "DIS"]

                if df_trades[today, "Shares"] == -1000:
                    df_trades[today, "Trades"] = 2000
                    df_trades[tomorrow, "Shares"] = 1000
                    df_trades[tomorrow, "Cash"] = df_trades[tomorrow, "Shares"]*df_trades[tomorrow, "DIS"]
                else:
                    df_trades[today, "Trades"] = 0
                    df_trades[tomorrow, "Shares"] = 1000
                    df_trades[tomorrow, "Cash"] = df_trades[tomorrow, "Shares"]*df_trades[tomorrow, "DIS"]

            if data[today, "DIS"] > data[tomorrow, "DIS"]:
                if df_trades[today, "Shares"] == 0:
                    df_trades[today, "Trades"] = -1000
                    df_trades[tomorrow, "Shares"] = -1000
                    df_trades[tomorrow, "Cash"] = df_trades[tomorrow, "Shares"]*df_trades[tomorrow, "DIS"]

                if df_trades[today, "Shares"] == 1000:
                    df_trades[today, "Trades"] = -2000
                    df_trades[tomorrow, "Shares"] = -1000
                    df_trades[tomorrow, "Cash"] = df_trades[tomorrow, "Shares"]*df_trades[tomorrow, "DIS"]
                else:
                    df_trades[today, "Trades"] = 0
                    df_trades[tomorrow, "Shares"] = -1000
                    df_trades[tomorrow, "Cash"] = df_trades[tomorrow, "Shares"]*df_trades[tomorrow, "DIS"]

                    

        return df_trades["Trades"]

class TechnicalStrategy:
    def __init__(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def train(self, *params, **kwparams):
        # Defined so you can call it with any parameters and it will just do nothing.
        pass

    def test(start_date = '2018-01-01', end_date = '2019-12-31', symbol = 'DIS', starting_cash = 200000):
        # Inputs represent the date range to consider, the single stock to trade, and the starting portfolio value.
        #
        # Return a date-indexed DataFrame with a single column containing the desired trade for that date.
        # Given the position limits, the only possible values are -2000, -1000, 0, 1000, 2000.


        data = get_data(
            start_date,
            end_date,
            [symbol],
            column_name="Adj Close",
            include_spy=False,
        )

        calc_aroon(data, 5)
        calc_BB(data, 5)
        calc_ema(data, 5)

        today = data.index[i]

        aroon = data[today, "Aroon"]
        ema = data[today, "Price/EMA"]
        bb = data[today, "BB%"]

        df_trades = data.copy()
        df_trades.drop(columns=["DIS"])
        df_trades["Trade"] = np.NaN
        df_trades["Cash"] = 200000
        df_trades["Shares"] = 0
        


        for i in range(5, data.shape[0]):
            tomorrow = df_trades.index[i + 1]
            if bb <= 0 and (aroon >= 25 or aroon == "NaN") and ema < 0.95: 
                    df_trades[tomorrow, "Shares"] = 1000

            if bb >= 1 and (aroon <= -25 or aroon == "NaN") and ema > 1.05: 
                    df_trades[tomorrow, "Shares"] = -1000
            
            else: 
                df_trades[tomorrow, "Shares"] = 0

            
            






        
        
        
        return df_trades