# TabularQLearnerEP.py
# Evan Phillips and Sumer Vaidya
# CSCI 3465

import numpy as np
import pandas as pd
import random


class TabularQLearnerEP:
    def __init__(
        self,
        states=100,
        actions=4,
        alpha=0.2,
        gamma=0.9,
        epsilon=0.98,
        epsilon_decay=0.999,
        dyna=0,
    ):

        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna
        self.q_table = np.random.uniform(low=-0.01, high=0.01, size=(states, actions))
        self.prev_state = None
        self.prev_action = None
        self.experience_tuples = []

    def train(self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # used to move the robot around the world

        best_action = self.get_action(s)

        old_q = (1 - self.alpha) * self.q_table[self.prev_state][self.prev_action]
        new_q = self.alpha * (r + (self.gamma * self.q_table[s][best_action]))
        self.q_table[self.prev_state][self.prev_action] = old_q + new_q

        experience_tuple = (self.prev_state, self.prev_action, s, r)
        self.experience_tuples.append(experience_tuple)

        #### DYNA-Q ######
        for i in range(self.dyna):
            # pick random indice
            rand_indice = random.randint(0, len(self.experience_tuples) - 1)
            rand_exp = self.experience_tuples[rand_indice]
            last_state, last_action, new_state, reward = (
                rand_exp[0],
                rand_exp[1],
                rand_exp[2],
                rand_exp[3],
            )

            best_action_dq = self.get_action(new_state)

            old_q_dq = (1 - self.alpha) * self.q_table[last_state][last_action]
            new_q_dq = self.alpha * (
                reward + (self.gamma * self.q_table[new_state][best_action_dq])
            )
            self.q_table[last_state][last_action] = old_q_dq + new_q_dq

        # add element of randomness (epsilon-greedy)
        if np.random.random() < self.epsilon:
            best_action = random.randint(0, self.actions - 1)
            self.epsilon *= self.epsilon_decay

        self.prev_state = s
        self.prev_action = best_action

        return best_action

    def test(self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.

        best_action = self.get_action(s)
        if self.prev_state is None and self.prev_action is None:
            self.prev_state = s
            self.prev_action = best_action

        self.prev_state = s

        return best_action

    def get_action(self, new_state):
        # find best action
        max_q = -100000
        best_action = None
        actions = list(range(self.actions))
        for act in actions:
            if self.q_table[new_state][act] > max_q:
                max_q = self.q_table[new_state][act]
                best_action = act

        return best_action
