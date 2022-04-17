# OracleStrategy.py
# CSCI 3465
# Evan Phillips and Sumer Vaidya

from BacktestEP import *
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
        data["Daily Return"] = abs(data["DIS"].shift(periods=1) - data["DIS"])

        for i in range(df_trades.shape[0] - 1):
            today = df_trades.index[i]
            tomorrow = df_trades.index[i + 1]

            if data.loc[today, "DIS"] <= data.loc[tomorrow, "DIS"]:
                df_trades.loc[today, "Direction"] = "BUY"
                df_trades.loc[today, "Shares"] = 1000

            if data.loc[today, "DIS"] > data.loc[tomorrow, "DIS"]:
                df_trades.loc[today, "Direction"] = "SELL"
                df_trades.loc[today, "Shares"] = -1000

        df_trades["Trades"] = abs(df_trades["Shares"].shift() - df_trades["Shares"])

        start = df_trades.index[0]
        df_trades.loc[start, "Trades"] = df_trades.loc[start, "Shares"]

        # ------------------------------------------------------------------- #
        df_trades = df_trades.drop(columns=["Shares"])
        df_trades.reset_index(inplace=True)
        df_trades = df_trades.rename(columns={"index": "Date"})
        df_trades = df_trades.rename(columns={"Trades": "Shares"})

        return df_trades

    def plot(trades):
        # PLOT CUMULATIVE RETURNS
        port = assess_strategy(trades, False)
        baseline = assess_strategy(trades, True)

        port["Cumulative Portfolio Returns"] = (
            port["Portfolio Value"] / port["Portfolio Value"][0] - 1
        )

        baseline["Cumulative Portfolio Returns"] = (
            baseline["Portfolio Value"] / baseline["Portfolio Value"][0] - 1
        )

        plt.plot(
            port["Cumulative Portfolio Returns"],
            label="Oracle Cumulative Portfolio Returns",
        )
        plt.plot(
            baseline["Cumulative Portfolio Returns"],
            label="Baseline Cumulative Portfolio Returns",
        )

        plt.legend()
        plt.grid()
        plt.title("Oracle vs Baseline Strategy")
        plt.xlabel("Date")
        plt.ylabel("Return (zero-based)")
        plt.show()

    trades = test()
    plot(trades)
    port = assess_strategy(trades, False)
    strategy_stats(port, "^SPX", trades)
