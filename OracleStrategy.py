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
        df_trades = df_trades.drop(columns=["DIS"])
        df_trades["Symbol"] = "DIS"
        df_trades["Direction"] = np.NaN
        df_trades["Trades"] = np.NaN
        df_trades["Shares"] = 0
        data["Daily Return"] = abs(data["DIS"]-data["DIS"].shift())

        for i in range(df_trades.shape[0]-1):
            today = df_trades.index[i]
            tomorrow = df_trades.index[i + 1]
            
            if data.loc[today, "DIS"] < data.loc[tomorrow, "DIS"]:
                if df_trades.loc[today, "Shares"] == 0:

                    df_trades.loc[today, "Trades"] = 1000
                    df_trades.loc[tomorrow, "Shares"] = 1000
                    df_trades.loc[today, "Direction"] = "BUY"

                elif df_trades.loc[today, "Shares"] == -1000:
                    df_trades.loc[today, "Trades"] = 2000
                    df_trades.loc[tomorrow, "Shares"] = 1000
                    df_trades.loc[today, "Direction"] = "BUY"

                else:
                    df_trades.loc[today, "Trades"] = 0
                    df_trades.loc[tomorrow, "Shares"] = 1000


            if data.loc[today, "DIS"] > data.loc[tomorrow, "DIS"]:
                if df_trades.loc[today, "Shares"] == 0:
                    df_trades.loc[today, "Trades"] = 1000
                    df_trades.loc[tomorrow, "Shares"] = -1000
                    df_trades.loc[today, "Direction"] = "SELL"


                elif df_trades.loc[today, "Shares"] == 1000:
                    df_trades.loc[today, "Trades"] = 2000
                    df_trades.loc[tomorrow, "Shares"] = -1000
                    df_trades.loc[today, "Direction"] = "SELL"

                else:
                    df_trades.loc[today, "Trades"] = 0
                    df_trades.loc[today, "Shares"] = -1000

        print("+++++++++++++++++++")
        print(df_trades.drop(columns=["Shares"]))
        
        return df_trades.drop(columns=["Shares"])


    test()