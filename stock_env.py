# stock_env.py
# Evan Phillips and Sumer Vaidya
# CSCI 3465

from operator import indexOf
from random import randint
import numpy as np
import pandas as pd
import argparse
from TabularQLearnerEP import *
from tech_ind import *
from TabularQLearner import *
from BacktestEP import *


class StockEnvironment:
    def __init__(self, fixed=None, floating=None, starting_cash=None, share_limit=None):
        self.shares = share_limit
        self.fixed_cost = fixed
        self.floating_cost = floating
        self.starting_cash = starting_cash
        self.indices = []
        self.q_learner = None
        self.aroon_list = None
        self.ema_list = None
        self.bb_list = None

        self.aroon_len = 0
        self.ema_len = 0
        self.bb_len = 0


    def prepare_world(self, start_date, end_date, symbol, data_folder=None):
        """
        Read the relevant price data and calculate some indicators.
        Return a DataFrame containing everything you need.
        """
        data = calc_ema(start_date, end_date, [symbol], 10)
        data = calc_aroon(data, 20)
        data = calc_BB(data, 10)
        data["Daily Return"] = data[symbol] / data[symbol].shift() - 1


        self.aroon_list = data["Aroon Oscillator"].tolist()[20:]
        self.ema_list = data["Price/EMA"].tolist()[10:]
        self.bb_list = data["BB%"].tolist()[10:]

        self.bb_list.sort()
        self.aroon_list.sort()
        self.ema_list.sort()

        self.aroon_len = len(self.aroon_list)
        self.ema_len = len(self.ema_list)
        self.bb_len = len(self.bb_list)

        return data

    def calc_state(self, df, day, holdings):
        """Quantizes the state to a single number."""
        today_aroon = df.loc[day, "Aroon Oscillator"]
        today_ema = df.loc[day, "Price/EMA"]
        today_bb = df.loc[day, "BB%"]

        if today_bb <= self.bb_list[int((1/5)*self.bb_len)]: 
            bb_val = 1 
        elif today_bb <= self.bb_list[int((2/5)*self.bb_len)]: 
            bb_val = 2 
        elif today_bb <= self.bb_list[int((3/5)*self.bb_len)]: 
            bb_val = 3
        elif today_bb <= self.bb_list[int((4/5)*self.bb_len)]: 
            bb_val = 4    
        else: 
            bb_val = 5


        if today_aroon <= self.aroon_list[int((1/5)*self.aroon_len)]: 
            aroon_val = 1 
        elif today_aroon <= self.aroon_list[int((2/5)*self.aroon_len)]: 
            aroon_val = 2 
        elif today_aroon <= self.aroon_list[int((3/5)*self.aroon_len)]: 
            aroon_val = 3
        elif today_aroon <= self.aroon_list[int((4/5)*self.aroon_len)]: 
            aroon_val = 4
        else: 
            aroon_val = 5
            
        if today_ema < self.ema_list[int((1/5)*self.ema_len)]: 
            ema_val = 1 
        elif today_ema < self.ema_list[int((2/5)*self.ema_len)]: 
            ema_val = 2 
        elif today_ema < self.ema_list[int((3/5)*self.ema_len)]: 
            ema_val = 3 
        elif today_ema < self.ema_list[int((4/5)*self.ema_len)]: 
            ema_val = 4 
        else: 
            ema_val = 5
            
        if holdings == -1000:
            pos_val = 1
        elif holdings == 0:
            pos_val = 2
        else:
            pos_val = 3

        quant = str(aroon_val) + str(ema_val) + str(bb_val) + str(pos_val)
        # put quantized values into array
        if quant in self.indices:
            return (self.indices.index(quant), quant)
        else:
            self.indices.append(quant)
            return (self.indices.index(quant), quant)

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


        self.q_learner = TabularQLearnerEP(states=375, actions=3, epsilon=eps, epsilon_decay=eps_decay, dyna=dyna)
        my_world = self.prepare_world(start, end, symbol)

        for j in range(500):
            my_world["Cash"] = self.starting_cash
            my_world["Reward"] = 0
            my_world["Portfolio"] = self.starting_cash
            my_world["Holdings"] = 0
            my_world["Shares"] = 0
            my_world["State"] = 0
            my_world["Direction"] = None
            print(j)

            first_state = self.calc_state(my_world, my_world.index[19], 0)[0]
            action = self.q_learner.test(first_state)
            holdings = self.action_to_holding(action)

            for i in range(20, my_world.shape[0]):
                today = my_world.index[i]
                yesterday = my_world.index[i - 1]

                my_world.loc[today, "Holdings"] = holdings
                current_state = self.calc_state(my_world, today, holdings)[0]
                my_world.loc[today, "State"] = self.calc_state(my_world, today, holdings)[1]
                
                trade = (my_world.loc[today, "Holdings"]-my_world.loc[yesterday, "Holdings"])

                trans_val = trade*my_world.loc[yesterday, symbol] 
                cost =  self.fixed_cost +  self.floating_cost*abs(trans_val)

                if trade == 0:
                    my_world.loc[yesterday, "Shares"] = 0
                    my_world.loc[today, "Cash"] = my_world.loc[yesterday, "Cash"]
                    my_world.loc[yesterday, "Direction"] = "NONE"

                if trade < 0:
                    my_world.loc[today, "Cash"] = my_world.loc[yesterday, "Cash"] - trans_val - cost 
                    my_world.loc[yesterday, "Shares"] = abs(trade)
                    my_world.loc[yesterday, "Direction"] = "SELL"

                if trade > 0:
                    my_world.loc[today, "Cash"] = my_world.loc[yesterday, "Cash"] - trans_val - cost 
                    my_world.loc[yesterday, "Shares"] = trade
                    my_world.loc[yesterday, "Direction"] = "BUY"

                
                today_port_val = my_world.loc[today, "Cash"] + my_world.loc[today, "Holdings"]*my_world.loc[today, symbol]
                yesterday_port_val = my_world.loc[yesterday, "Cash"] + my_world.loc[yesterday, "Holdings"]*my_world.loc[yesterday, symbol]
                
                reward = today_port_val/yesterday_port_val - 1
                my_world.loc[yesterday, "Reward"] = reward
                my_world.loc[today, "Portfolio"] = today_port_val
                new_action = self.q_learner.train(current_state, reward)
                holdings = self.action_to_holding(new_action)

        trades = my_world[["Direction", "Shares", "Holdings",symbol, "Reward", "State", "Portfolio"]]
        trades["Symbol"] = symbol
        trades.reset_index(inplace=True)
        trades = trades.rename(columns={"index": "Date"})

        print(trades.to_string())
        return trades

    def test_learner(self, start=None, end=None, symbol=None):

        """
        Evaluate a trained Q-Learner on a particular stock trading task.

        Print a summary result of what happened during the test.
        Feel free to include portfolio stats or other information, but AT LEAST:

        Test trip, net result: $31710.00
        Benchmark result: $6690.0000
        """

        my_world = self.prepare_world(start, end, symbol)
        # my_world = self.prepare_world("2018-01-01", "2018-03-31", "DIS")
        # my_world["Shares"] = 0
        # my_world["Direction"] = 0
        # my_world["Holdings"] = 0
        # first_state = self.calc_state(my_world, my_world.index[19], 0)
        # action = self.q_learner.test(first_state)
        # holdings = self.action_to_holding(action)
        # for i in range(20, my_world.shape[0] - 1):
        #     today = my_world.index[i]
        #     yesterday = my_world.index[i - 1]

        #     current_state = self.calc_state(my_world, today, holdings)
        #     my_world.loc[today, "Holdings"] = holdings

        #     trade = (
        #         my_world.loc[today, "Holdings"]
        #         - my_world.loc[yesterday, "Holdings"]
        #     )

        #     if trade == 0:
        #         my_world.loc[yesterday, "Shares"] = 0
        #         my_world.loc[yesterday, "Direction"] = "NONE"

        #     if trade < 0:
        #         my_world.loc[yesterday, "Shares"] = abs(trade)
        #         my_world.loc[yesterday, "Direction"] = "SELL"

        #     if trade > 0:
        #         my_world.loc[yesterday, "Shares"] = trade
        #         my_world.loc[yesterday, "Direction"] = "BUY"

        #     new_action = self.q_learner.test(current_state)
        #     holdings = self.action_to_holding(new_action)

        # trades = my_world[["Shares", "Direction"]]
        # trades["Symbol"] = symbol
        # trades.reset_index(inplace=True)
        # trades = trades.rename(columns={"index": "Date"})
        # # exit()
        # port = assess_strategy(trades, False)
        # strategy_stats(port, symbol, trades)

        # return trades
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
        default="2018-02-10",
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
    trades = env.train_learner(
        start=args.train_start,
        end=args.train_end,
        symbol=args.symbol,
        trips=args.trips,
        dyna=args.dyna,
        eps=args.eps,
        eps_decay=args.eps_decay,
    )

    port = assess_strategy(trades, False, fixed_cost = 0, floating_cost= 0)
    strategy_stats(port, args.symbol, trades)

    # Test the learned policy and see how it does.

    # In sample.
    env.test_learner(start=args.train_start, end=args.train_end, symbol=args.symbol)

    # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
    # env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )
