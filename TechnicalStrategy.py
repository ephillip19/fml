# TechnicalStrategy.py
# CSCI 3465
# Evan Phillips and Sumer Vaidya

from BacktestEP import *
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
        df_trades["Symbol"] = "DIS"
        df_trades["Direction"] = np.NaN
        df_trades["Trades"] = np.NaN
        df_trades["Shares"] = 0

        data = calc_aroon(data, 20)
        data = calc_BB(data, 5)
        data = calc_ema(data, 10)

        trade_dates = []

        for i in range(5, data.shape[0] - 1):
            today = data.index[i]
            yesterday = data.index[i - 1]

            aroon = data.loc[today, "Aroon Oscillator"]
            ema = data.loc[today, "Price/EMA"]
            bb = data.loc[today, "BB%"]

            # identify BUY/SELL oppertunity for a good long term holding

            # first buy around 2018-04
            if (
                bb < 0.17 and (aroon > 40 or aroon == "nan") and ema < 0.98
            ) and df_trades.loc[today, "Shares"] != 1000:
                df_trades.loc[today:end_date, "Shares"] = 1000
                df_trades.loc[today:end_date, "Direction"] = "BUY"

            # first Sell around 2018-01
            elif (
                bb > 0.85 and (aroon < -40 or aroon == "nan") and ema > 1.02
            ) and df_trades.loc[today, "Shares"] != -1000:
                df_trades.loc[today:end_date, "Shares"] = -1000
                df_trades.loc[today:end_date, "Direction"] = "SELL"

            # TRADING ON STRONG MOMENTUM
            elif aroon >= 95:
                df_trades.loc[today:end_date, "Shares"] = 1000
                df_trades.loc[today:end_date, "Direction"] = "BUY"

            elif aroon <= -95:
                df_trades.loc[today:end_date, "Shares"] = 1000
                df_trades.loc[today:end_date, "Direction"] = "BUY"

            # #identify uncertainity and hold risk averse position
            elif (-40 < aroon <= 40) and (0.4 <= bb <= 0.6) and (0.99 < ema < 1.012):

                if df_trades.loc[yesterday, "Shares"] == 1000:
                    df_trades.loc[today:end_date, "Direction"] = "SELL"
                    df_trades.loc[today:end_date, "Shares"] = 0

                elif df_trades.loc[yesterday, "Shares"] == -1000:
                    df_trades.loc[today:end_date, "Direction"] = "BUY"
                    df_trades.loc[today:end_date, "Shares"] = 0

        df_trades["Trades"] = abs(df_trades["Shares"] - df_trades["Shares"].shift())
        start = df_trades.index[0]
        df_trades.loc[start, "Trades"] = df_trades.loc[start, "Shares"]

        # get dates on which a trade was made
        for i in range(0, df_trades.shape[0] - 1):
            today = df_trades.index[i]
            if df_trades.loc[today, "Trades"] != 0:
                trade_dates.append((today, df_trades.loc[today, "Direction"]))

        ####### CLEANING-UP ################
        df_trades = df_trades.drop(columns=["Shares"])
        df_trades.reset_index(inplace=True)
        df_trades = df_trades.rename(columns={"index": "Date"})
        df_trades = df_trades.rename(columns={"Trades": "Shares"})

        # OPTIONAL TO RETURN trade_dates IN ORDER TO ACCESS THEM IN plot()
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
            label="Our Technical Strategy Cumulative Portfolio Returns",
        )
        plt.plot(
            baseline["Cumulative Portfolio Returns"],
            label="Baseline Cumulative Portfolio Returns",
        )
        # for trade in dates:
        #     if trade[1] == "BUY":
        #         plt.axvline(x=trade[0], color="g", linewidth=0.5)
        #     else:
        #         plt.axvline(x=trade[0], color="r", linewidth=0.5)
        plt.legend()
        plt.title("Our Technical Strategy vs Baseline Strategy")
        plt.xlabel("Date")
        plt.ylabel("Return (zero-based)")
        plt.show()

    # run = test()
    # trades = run[0]
    # dates = run[1]
    trades = test()
    plot(trades)
    port = assess_strategy(trades, False)
    strategy_stats(port, "^SPX", trades)
