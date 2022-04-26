# stock_env.py
# Evan Phillips and Sumer Vaidya
# CSCI 3465

from operator import indexOf
import numpy as np
import pandas as pd
import argparse
from tech_ind import *
from TabularQLearnerEP import *


class StockEnvironment:
    def __init__(self, fixed=None, floating=None, starting_cash=None, share_limit=None):
        self.shares = share_limit
        self.fixed_cost = fixed
        self.floating_cost = floating
        self.starting_cash = starting_cash
        self.indices = []

    def prepare_world(self, start_date, end_date, symbol, data_folder=None):
        """
        Read the relevant price data and calculate some indicators.
        Return a DataFrame containing everything you need.
        """
        data = calc_ema(start_date, end_date, [symbol], 25)
        data = calc_aroon(data, 25)
        data = calc_BB(data, 30)
        data["Daily Return"] = data[symbol] / data[symbol].shift() - 1
        return data

    def calc_state(self, df, day, holdings):
        """Quantizes the state to a single number."""

        coef = int(df.shape[0] / 3)

        aroon_list = df["Aroon Oscillator"].tolist()[25:]
        ema_list = df["Price/EMA"].tolist()[25:]
        bb_list = df["BB%"].tolist()[30:]

        bb_list.sort()
        aroon_list.sort()
        ema_list.sort()

        today_aroon = df.loc[day, "Aroon Oscillator"]
        today_ema = df.loc[day, "Price/EMA"]
        today_bb = df.loc[day, "BB%"]

        pos_val = 0
        aroon_val = 0
        ema_val = 0
        bb_val = 0

        for i in range(1, 3):

            if aroon_list[coef * (i - 1)] <= today_aroon <= aroon_list[coef * i]:
                aroon_val = i

            if ema_list[coef * (i - 1)] <= today_ema <= ema_list[coef * i]:
                ema_val = i

            if bb_list[coef * (i - 1)] <= today_bb <= bb_list[coef * i]:
                bb_val = i

        if aroon_val == 0:
            aroon_val == 3
        if ema_val == 0:
            ema_val == 3
        if bb_val == 0:
            bb_val == 3

        if holdings == -1000:
            pos_val = 1
        elif holdings == 0:
            pos_val = 2
        else:
            pos_val = 3

        quant = str(aroon_val) + str(ema_val) + str(bb_val) + str(pos_val)

        # put quantized values into array
        if quant in self.indices:
            return self.indices.index(quant)
        else:
            self.indices.append(quant)
            return self.indices.index(quant)

        # return int(quant)

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

        my_world = self.prepare_world("2018-01-01", "2019-12-31", "DIS")
        my_world["Position"] = 0
        my_world["Shares"] = 0
        my_world["Direction"] = 0
        q_learner = TabularQLearnerEP(states=81, actions=3)

        print("here")

        for j in range(1):

            first_state = self.calc_state(my_world, my_world.index[30], 0)
            action = q_learner.test(first_state)
            holdings = self.action_to_holding(action)

            for i in range(31, my_world.shape[0] - 1):
                today = my_world.index[i]
                yesterday = my_world.index[i - 1]

                
                current_state = self.calc_state(my_world, today, holdings)

                my_world.loc[today, "Position"] = holdings
                trade = (
                    my_world.loc[today, "Position"] - my_world.loc[yesterday, "Position"]
                )
            

                if trade == 0:
                    reward = holdings * my_world.loc[today, "Daily Return"]
                    my_world.loc[yesterday, "Shares"] = 0

                if trade < 0:
                    reward = (
                        holdings * my_world.loc[today, "Daily Return"]
                        - self.fixed_cost
                        + trade * self.floating_cost
                    )
                    
                    my_world.loc[yesterday, "Shares"] = 1000
                    my_world.loc[yesterday, "Direction"] = "SELL"

                if trade > 0:
                    reward = (
                        holdings * my_world.loc[today, "Daily Return"]
                        - self.fixed_cost
                        - trade * self.floating_cost
                    )
                    
                    my_world.loc[yesterday, "Shares"] = 1000
                    my_world.loc[yesterday, "Direction"] = "BUY"

                new_action = q_learner.train(current_state, reward)
                holdings = self.action_to_holding(new_action)

        return my_world[[symbol, "Shares", "Direction" ]]

    def test_learner(self, start=None, end=None, symbol=None):
        """
        Evaluate a trained Q-Learner on a particular stock trading task.

        Print a summary result of what happened during the test.
        Feel free to include portfolio stats or other information, but AT LEAST:

        Test trip, net result: $31710.00
        Benchmark result: $6690.0000
        """
        pass

    def action_to_holding(self, action):
        if action == 0:
            return -1000
        elif action == 1:
            return 0
        else:
            return 1000


if __name__ == "__main__":
    # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
    # and let it start trading.

    parser = argparse.ArgumentParser(description="Stock environment for Q-Learning.")

    date_args = parser.add_argument_group("date arguments")
    date_args.add_argument(
        "--train_start",
        default="2018-01-01",
        metavar="DATE",
        help="Start of training period.",
    )
    date_args.add_argument(
        "--train_end",
        default="2019-12-31",
        metavar="DATE",
        help="End of training period.",
    )
    date_args.add_argument(
        "--test_start",
        default="2020-01-01",
        metavar="DATE",
        help="Start of testing period.",
    )
    date_args.add_argument(
        "--test_end",
        default="2021-12-31",
        metavar="DATE",
        help="End of testing period.",
    )

    learn_args = parser.add_argument_group("learning arguments")
    learn_args.add_argument(
        "--dyna", default=0, type=int, help="Dyna iterations per experience."
    )
    learn_args.add_argument(
        "--eps",
        default=0.99,
        type=float,
        metavar="EPSILON",
        help="Starting epsilon for epsilon-greedy.",
    )
    learn_args.add_argument(
        "--eps_decay",
        default=0.99995,
        type=float,
        metavar="DECAY",
        help="Decay rate for epsilon-greedy.",
    )

    sim_args = parser.add_argument_group("simulation arguments")
    sim_args.add_argument(
        "--cash", default=200000, type=float, help="Starting cash for the agent."
    )
    sim_args.add_argument(
        "--fixed", default=0.00, type=float, help="Fixed transaction cost."
    )
    sim_args.add_argument(
        "--floating", default="0.00", type=float, help="Floating transaction cost."
    )
    sim_args.add_argument(
        "--shares",
        default=1000,
        type=int,
        help="Number of shares to trade (also position limit).",
    )
    sim_args.add_argument("--symbol", default="DIS", help="Stock symbol to trade.")
    sim_args.add_argument(
        "--trips", default=500, type=int, help="Round trips through training data."
    )

    args = parser.parse_args()

    # Create an instance of the environment class.
    env = StockEnvironment(
        fixed=args.fixed,
        floating=args.floating,
        starting_cash=args.cash,
        share_limit=args.shares,
    )

    # Construct, train, and store a Q-learning trader.
    env.train_learner(
        start=args.train_start,
        end=args.train_end,
        symbol=args.symbol,
        trips=args.trips,
        dyna=args.dyna,
        eps=args.eps,
        eps_decay=args.eps_decay,
    )

    # Test the learned policy and see how it does.

    # In sample.
    env.test_learner(start=args.train_start, end=args.train_end, symbol=args.symbol)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    # env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )
