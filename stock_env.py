# stock_env.py
# Evan Phillips and Sumer Vaidya
# CSCI 3465

import numpy as np
import pandas as pd
from tech_ind import *


class StockEnvironment:
    def __init__(self, fixed=None, floating=None, starting_cash=None, share_limit=None):
        self.shares = share_limit
        self.fixed_cost = fixed
        self.floating_cost = floating
        self.starting_cash = starting_cash

    def prepare_world(self, start_date, end_date, symbol, data_folder):
        """
        Read the relevant price data and calculate some indicators.
        Return a DataFrame containing everything you need.
        """
        data = calc_ema(start_date, end_date, [symbol], 25)
        data = calc_aroon(data, 25)
        data = calc_BB(data, 30)

        return data

    def calc_state(self, df, day, holdings):
        """Quantizes the state to a single number."""
        pass

    def train_learner(
        self, start=None, end=None, symbol=None, trips=0, dyna=0, eps=0.0, eps_decay=0.0
    ):
        """
        Construct a Q-Learning trader and train it through many iterations of a stock
        world.  Store the trained learner in an instance variable for testing.

        Print a summary result of what happened at the end of each trip.
        Feel free to include portfolio stats or other information, but AT LEAST:

        Trip 499 net result: $13600.00
        """
        pass

    def test_learner(self, start=None, end=None, symbol=None):
        """
        Evaluate a trained Q-Learner on a particular stock trading task.
        Print a summary result of what happened during the test.

        Feel free to include portfolio stats or other information, but AT LEAST:

        Test trip, net result: $31710.00
        Benchmark result: $6690.0000
        """
        pass
