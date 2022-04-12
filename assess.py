# assess.py
# Evan Zamora Phillips
# CSCI 3465

import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def main():

    stats = assess_portfolio(
        "2021-01-11",
        "2021-12-20",
        ["NFLX", "AMZN", "XOM", "PTON"],
        [0.0, 0.35, 0.35, 0.3],
        plot_returns=True,
        starting_value=500000,
        sample_freq=52,
    )

    # PRINT OUTPUTS
    # print("Sharpe Ration:", stats[3])
    # print("Volatility (stdev of daily returns):", stats[2])
    # print("Average Daily Return:", stats[1])
    # print("Cumulative Return:", stats[0])
    # print("Ending Value:", stats[4])


def get_data(start_date, end_date, symbols, column_name="Adj Close", include_spy=False):
    dates = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(index=dates)
    df_new = pd.read_csv(
        "data/SPY.csv",
        index_col="Date",
        parse_dates=True,
        usecols=["Date", column_name],
    )
    data = df.join(df_new, how="inner")
    data = data.rename(columns={column_name: "SPY"})

    if not include_spy:
        data = data.drop(columns=["SPY"])

    for symbol in symbols:
        df_stock = pd.read_csv(
            "data/" + symbol + ".csv",
            index_col="Date",
            parse_dates=True,
            usecols=["Date", column_name],
        )
        data = data.join(df_stock, how="left")
        data = data.rename(columns={column_name: symbol})

    return data


def assess_portfolio(
    start_date,
    end_date,
    symbols,
    allocations,
    starting_value=1000000,
    risk_free_rate=0.0,
    sample_freq=252,
    plot_returns=False,
):

    data = get_data(start_date, end_date, symbols, include_spy=True)
    data /= data.iloc[0]
    stocks = data.iloc[:, 1:]
    stocks *= allocations
    stocks *= starting_value
    data["Total"] = stocks.sum(axis=1)
    data["Daily Return"] = data["Total"] / data["Total"].shift() - 1
    data["Cumulative Portfolio Returns"] = data["Total"] / data["Total"][0] - 1
    data["Cumulative SPY Returns"] = data["SPY"] / data["SPY"][0] - 1

    # OUTPUTS
    end_value = data["Total"][-1]
    cumulative_return = data["Cumulative Portfolio Returns"][-1]
    average_daily_return = data["Daily Return"].mean()
    stdev_daily_return = data["Daily Return"].std()
    sharpe_ratio = (
        math.sqrt(sample_freq)
        * (average_daily_return - risk_free_rate)
        / stdev_daily_return
    )

    if plot_returns:
        plt.plot(data["Cumulative SPY Returns"], label="SPY")
        plt.plot(data["Cumulative Portfolio Returns"], label="Portfolio")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.title("Daily portfolio value and SPY")
        plt.grid()
        plt.show()

    return (
        cumulative_return,
        average_daily_return,
        stdev_daily_return,
        sharpe_ratio,
        end_value,
    )


def get_adr(port, col):
    return np.mean(port[col])


def get_stdev(port, col):
    return (port[col]).std()


def get_sr(port, adr, stdev, sample_freq=252, risk_free_rate=0.0):
    return math.sqrt(sample_freq) * (adr - risk_free_rate) / stdev


def get_cr(port, col):
    return port[col][-1]


if __name__ == "__main__":
    main()
