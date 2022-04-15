from assess import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tech_ind import *



class TechnicalStrategy:
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
        df_trades = df_trades.drop(columns=[symbol])
        df_trades["Symbol"] = symbol
        df_trades["Direction"] = np.NaN
        df_trades["Shares"] = 0

        data = calc_aroon(data, 25)
        data = calc_BB(data, 5)
        data = calc_ema(data, 10)


        for i in range(10, data.shape[0]-1):
            today = data.index[i]
            tomorrow = df_trades.index[i + 1]

            aroon = data.loc[today, "Aroon Oscillator"]
            ema = data.loc[today, "Price/SMA"]
            bb = data.loc[today, "BB%"]

            print((bb, aroon, ema))
            print("------")

            if bb <= 0.5 and (aroon >= 25 or aroon == "NaN") and ema < 0.95:
                df_trades.loc[tomorrow, "Shares"] = 1000
                df_trades.loc[tomorrow, "Direction"] = "BUY"
            if bb >= 0.5 and (aroon <= -25 or aroon == "NaN") and ema > 1.05:
                df_trades.loc[tomorrow, "Shares"] = -1000
                df_trades.loc[tomorrow, "Direction"] = "BUY"

            else:
                df_trades.loc[tomorrow, "Shares"] = 0
        
        
        df_trades["Trades"] = abs(df_trades["Shares"] - df_trades["Shares"].shift())

        return df_trades.drop(columns=["Shares"])
    print(test()["Trades"].sum())