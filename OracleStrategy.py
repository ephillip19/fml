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

        data["Cash"] = starting_cash
        df_trades = data.copy()
        df_trades = df_trades.drop(columns=["DIS"])
        df_trades["Trades"] = np.NaN

        shares = 0
        for i in range(df_trades.shape[0] - 1):
            today = df_trades.index[i]
            tomorrow = df_trades.index[i + 1]
            if data.loc[today, "DIS"] < data.loc[tomorrow, "DIS"]:
                if shares == 0:
                    df_trades.loc[today, "Trades"] = 2000

                if shares == 1000:
                    df_trades.loc[today, "Trades"] = 1000

                if shares == -1000:
                    df_trades.loc[today, "Trades"] = 2000

                if shares == -2000:
                    df_trades.loc[today, "Trades"] = 2000

            if data.loc[today, "DIS"] > data.loc[tomorrow, "DIS"]:
                if shares == 0:
                    df_trades.loc[today, "Trades"] = -2000

                if shares == 1000:
                    df_trades.loc[today, "Trades"] = -2000

                if shares == -1000:
                    df_trades.loc[today, "Trades"] = -1000

                if shares == 2000:
                    df_trades.loc[today, "Trades"] = -2000

        print(data)
        print(df_trades)
        return df_trades

    test()
