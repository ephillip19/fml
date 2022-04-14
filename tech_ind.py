# tech_ind.py
# CSCI 3465
from cgi import test
from lib2to3.pgen2.token import PERCENT

from pyparsing import col
from assess import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# GITHUB INSTRUCTIONS
# git add -A
# git commit -m "message"
# git push

# If you need to pull without saving your local changes: git stash

data = get_data(
    "1/1/2018", "12/31/2019", ["DIS"], column_name="Adj Close", include_spy=False
)


def calc_sma(prices, n):
    prices["SMA"] = prices["DIS"]

    for i in range(1, n):
        prices["SMA"] = prices["SMA"] + prices["DIS"].shift(periods=i)

    prices["SMA"] = prices["SMA"] / n
    return prices


def calc_BB(prices, n):

    prices = calc_sma(prices, n)

    prices["BB_lower"] = prices["SMA"]
    prices["BB_upper"] = prices["SMA"]

    for i in range(n - 1, prices.shape[0]):
        start = prices.index[i - (n - 1)]
        end = prices.index[i]

        prices.loc[end, "BB_lower"] = (
            prices.loc[end, "BB_lower"] - 2 * prices.loc[start:end, "DIS"].std()
        )
        prices.loc[end, "BB_upper"] = (
            prices.loc[end, "BB_upper"] + 2 * prices.loc[start:end, "DIS"].std()
        )

    prices["BB%"] = (prices["DIS"] - prices["BB_lower"]) / (
        prices["BB_upper"] - prices["BB_lower"]
    )
    # PLOT INDICATOR
    fig, axs = plt.subplots(2)
    fig.suptitle("Bollinger Band Analysis")
    axs[0].plot(prices["SMA"], color="g")
    axs[0].plot(prices["BB_lower"], label="BB-" + str(n), color="g")
    axs[0].plot(prices["BB_upper"], color="g")
    axs[0].plot(prices["DIS"], label="DIS", color="b")

    axs[1].plot(prices["BB%"], label="BB%", color="m")
    axs[0].legend()
    axs[1].legend()
    axs[1].set(xlabel="Date")
    axs[0].grid()
    axs[1].grid()
    axs[0].set(ylabel="Price")
    axs[1].set(ylabel="BB%")
    plt.show()

    return prices


def calc_ema(prices, n):

    alpha = 2 / (n + 1)
    prices = calc_sma(prices, n)

    prices["EMA"] = prices["SMA"]

    for i in range(n, prices.shape[0]):
        date = prices.index[i]
        prev_date = prices.index[i - 1]
        prices.loc[date, "EMA"] = (
            alpha * prices.loc[date, "DIS"] + (1 - alpha) * prices.loc[prev_date, "EMA"]
        )

    prices["Price/SMA"] = prices["DIS"] / prices["SMA"]
    prices["Price/EMA"] = prices["DIS"] / prices["EMA"]

    # PLOT INDICATOR
    fig, axs = plt.subplots(2)
    fig.suptitle("Moving Average Analysis")
    axs[0].plot(prices["SMA"], label="SMA-" + str(n))
    axs[0].plot(prices["EMA"], label="EMA-" + str(n))
    axs[0].plot(prices["DIS"], label="DIS")

    axs[1].plot(prices["Price/SMA"], label="Price/SMA")
    axs[1].plot(prices["Price/EMA"], label="Price/EMA")

    axs[0].legend()
    axs[1].legend()
    axs[1].set(xlabel="Date")
    axs[0].grid()
    axs[1].grid()
    axs[0].set(ylabel="Price")
    axs[1].set(ylabel="Price/Moving Average")
    plt.show()

    return prices


def calc_aroon(prices, n):
    prices["Aroon Up"] = np.NaN
    prices["Aroon Down"] = np.NaN
    prices["Aroon Oscillator"] = np.NaN
    for i in range(n, prices.shape[0]):

        period_start = prices.index[i - n]
        period_end = prices.index[i]
        period_array = prices.loc[period_start:period_end, "DIS"]
        period_list = period_array.tolist()
        # aroon up
        period_high = max(period_list)
        period_high_index = period_list.index(period_high)
        period_since_up = n - (period_high_index + 1)
        aroon_up = 100 * (n - period_since_up) / n

        prices.loc[period_end, "Aroon Up"] = aroon_up

        # aroon down
        period_low = min(period_list)
        period_low_index = period_list.index(period_low)
        period_since_down = n - (period_low_index + 1)
        aroon_down = 100 * (n - period_since_down) / n

        prices.loc[period_end, "Aroon Down"] = aroon_down

        prices.loc[period_end, "Aroon Oscillator"] = aroon_up - aroon_down

    # PLOT INDICATORS
    fig, axs = plt.subplots(2)
    fig.suptitle("Aroon Indicator Analysis")
    # axs[0].plot(prices["Aroon Oscillator"], label="Aroon Oscillator", color="m")
    axs[0].plot(prices["DIS"], label="DIS", color="b")
    axs0 = axs[0].twinx()
    axs0.plot(prices["Aroon Oscillator"], label="Aroon Oscillator", color="m")

    axs[1].plot(prices["Aroon Up"], label="Aroon Up", color="g")
    axs[1].plot(prices["Aroon Down"], label="Aroon Down", color="r")

    axs[0].legend(loc="lower left")
    axs0.legend()
    axs[1].legend(bbox_to_anchor=(0, -0.3), loc="lower center")
    axs[1].set(xlabel="Date")
    axs[0].grid()
    axs[1].grid()
    axs[0].set(ylabel="Price")
    axs0.set(ylabel="Aroon")
    axs[1].set(ylabel="Aroon")
    plt.show()

    return prices


def test():

    # calc_ema(data, 5)
    # calc_aroon(data, 25)
    # calc_BB(data, 30)

    return data


test()
