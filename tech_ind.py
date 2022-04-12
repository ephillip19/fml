# tech_ind.py
# CSCI 3465
from assess import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

data = get_data(
    "1/1/2018", "12/31/2019", ["DIS"], column_name="Adj Close", include_spy=False
)


def calc_sma(prices, n):
    prices["SMA"] = prices["DIS"]

    for i in range(1, n):
        prices["SMA"] = prices["SMA"] + prices["DIS"].shift(periods=i)

    prices["SMA"] = prices["SMA"] / n
    return prices


def BB(prices, n):

    prices = calc_sma(prices, n)

    prices["BB_lower"] = prices["SMA"]
    prices["BB_upper"] = prices["SMA"]

    for i in range(n - 1, prices.shape[0]):
        start = prices.index[i - (n - 1)]
        end = prices.index[i]

        prices.loc[end, "BB_lower"] = (
            prices.loc[end, "BB_lower"] + 2 * prices.loc[start:end, "DIS"].std()
        )
        prices.loc[end, "BB_upper"] = (
            prices.loc[end, "BB_upper"] - 2 * prices.loc[start:end, "DIS"].std()
        )

    prices["BB%"] = (prices["DIS"] - prices["BB_lower"]) / (
        prices["BB_upper"] - prices["BB_lower"]
    )

    print(prices)


BB(data, 5)
