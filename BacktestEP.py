# BacktestEP.py
# CSCI 3465
# Evan Phillips and Sumer Vaidya

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from assess import *


def main():
    # the class in this file was removed for the purposes of ease in running accompanying files
    print("n/a")


def assess_strategy(
    trade_file,
    baseline,
    starting_value=200000,
    fixed_cost=9.95,
    floating_cost=0.005,
):
    if ".csv" in trade_file:
        df_trades = pd.read_csv(trade_file)
    else:
        if baseline:
            df_trades = trade_file
            df_baseline = df_trades.head(1)
            df_baseline = df_baseline.append(df_trades.iloc[-1])
            if df_baseline.loc[0, "Direction"] != "BUY":
                df_baseline.loc[0, "Direction"] = "BUY"
                df_baseline.loc[0, "Shares"] = 1000.0

            df_baseline.loc[df_trades.shape[0] - 1, "Direction"] = "SELL"
            df_trades = df_baseline
        else:
            df_trades = trade_file

    df_trades.reset_index(inplace=True)
    df_trades = df_trades.drop(columns=["index"])

    start_date = df_trades["Date"].iloc[0]
    end_date = df_trades["Date"].iloc[-1]
    symbols = []
    for symbol in df_trades["Symbol"]:
        if symbol not in symbols:
            symbols.append(symbol)

    price_data = get_data(
        start_date, end_date, symbols, column_name="Adj Close", include_spy=False
    )

    # keep same indices but clear rest of dataframe
    share_data = price_data.copy()
    share_data["Cash"] = starting_value
    for symbol in symbols:
        share_data[symbol] = 0

    # print(df_trades.to_string())
    for i in range(df_trades.shape[0]):
        date_index = df_trades["Date"][i]
        ticker = df_trades["Symbol"][i]
        direction = df_trades["Direction"][i]
        shares = df_trades["Shares"][i]
        if direction == "BUY":
            share_data.loc[date_index:, ticker] += shares
            cash_minus = price_data.loc[date_index, ticker] * shares
            share_data.loc[date_index:, "Cash"] -= cash_minus
            share_data.loc[date_index:, "Cash"] -= fixed_cost + (
                floating_cost * cash_minus
            )

        if direction == "SELL":
            share_data.loc[date_index:, ticker] -= shares
            cash_plus = price_data.loc[date_index, ticker] * shares
            share_data.loc[date_index:, "Cash"] += cash_plus
            share_data.loc[date_index:, "Cash"] -= fixed_cost + (
                floating_cost * cash_plus
            )

    cash = share_data["Cash"].copy()
    share_data = share_data.drop(columns=["Cash"])
    portfolio = np.multiply(share_data, price_data)
    portfolio["Cash"] = cash
    portfolio["Portfolio Value"] = portfolio.sum(axis=1)
    portfolio["Cumulative Portfolio Returns"] = (
        portfolio["Portfolio Value"] / portfolio["Portfolio Value"][0] - 1
    )

    return portfolio[["Portfolio Value"]]


def strategy_stats(portfolio, benchmark, trade_file):
    if ".csv" in trade_file:
        df_trades = pd.read_csv(trade_file)
    else:
        df_trades = trade_file
    start_date = df_trades["Date"].iloc[0]
    end_date = df_trades["Date"].iloc[-1]
    benchmark_val = get_data(
        start_date,
        end_date,
        [benchmark],
        column_name="Adj Close",
        include_spy=False,
    )
    portfolio[benchmark] = benchmark_val
    # caluculating cumulative daily returns
    portfolio["Cumulative " + benchmark + "Returns"] = (
        portfolio[benchmark] / portfolio[benchmark][0] - 1
    )
    portfolio["Cumulative Portfolio Returns"] = (
        portfolio["Portfolio Value"] / portfolio["Portfolio Value"][0] - 1
    )
    # calculating daily returns
    portfolio["Portfolio Daily Returns"] = (
        portfolio["Portfolio Value"] / portfolio["Portfolio Value"].shift() - 1
    )
    portfolio[benchmark + " Daily Returns"] = (
        portfolio[benchmark] / portfolio[benchmark].shift() - 1
    )

    portfolio_adr = get_adr(portfolio, "Portfolio Daily Returns")
    benchmark_adr = get_adr(portfolio, benchmark + " Daily Returns")

    portfolio_cr = get_cr(portfolio, "Cumulative Portfolio Returns")
    benchmark_cr = get_cr(portfolio, "Cumulative " + benchmark + "Returns")

    portfolio_std = get_stdev(portfolio, "Portfolio Daily Returns")
    benchmark_std = get_stdev(portfolio, benchmark + " Daily Returns")

    portfolio_sr = get_sr(portfolio, portfolio_adr, portfolio_std)
    benchmark_sr = get_sr(portfolio, benchmark_adr, benchmark_std)

    final_port_val = portfolio["Portfolio Value"][-1]

    print("\n")
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("\n")
    print("Portfolio SR:", portfolio_sr)
    print("Benchmark SR:", benchmark_sr)
    print("\n")
    print("Portfolio ADR:", portfolio_adr)
    print("Benchmark ADR:", benchmark_adr)
    print("\n")
    print("Portfolio CR:", portfolio_cr)
    print("Benchmark CR:", benchmark_cr)
    print("\n")
    print("Portfolio SD:", portfolio_std)
    print("Benchmark SD:", benchmark_std)
    print("\n")
    print("Final Portfolio Value:", final_port_val)

    # PLOT portfolio daily value

    return 0


# port = assess_strategy()
# strategy_stats(port, "^SPX", "./trades/simple.csv")
if __name__ == "__main__":
    main()
