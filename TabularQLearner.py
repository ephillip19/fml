import numpy as np
import matplotlib.pyplot as plt
import random

class TabularQLearner:
    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):
        # Store all the parameters as attributes (instance variables).
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.curr_state_action = [None, None] #list to hold previous state action pair
        self.dyna = dyna
        # Initialize any data structures you need.
        self.q_table = np.random.random((self.states,self.actions)) #2D array to hold q-values
        self.exp_list = [] # list of lists to hold experience tuples for dynaQ

    def train (self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        #           How will you know the previous state and action?


        column = self.curr_state_action[1] #previous state 
        row = self.curr_state_action[0] #previous action 

        #select next best action: 
        counter = -1 
        action = 0
        curr_value = -10000000

        for i in self.q_table[s-1]: 
            counter+=1
            if i >= curr_value: 
                curr_value = i
                action = counter
 
        #update q value with new state, reward, and best aciton
        self.q_table[row-1, column] = (1-self.alpha)*self.q_table[row-1, column] + self.alpha*(r + self.gamma*self.q_table[s-1, action])
        #update current state action pair
        self.curr_state_action = [s, action]
        #add to experiences list
        self.exp_list.append([row, column, s, r])


    #implentation of DynaQ
        for i in range(0, self.dyna):
            #random selection of experience tuple from list of experiences
            sample = self.exp_list[random.randint(0, len(self.exp_list)-1)]
            state = sample[0]
            my_action = sample[1]
            new_state = sample[2]
            reward = sample[3]


            #select next best action: 
            counter = -1 
            my_newaction = 0
            curr_value = -10000000
        
            for i in self.q_table[new_state-1]: 
                counter+=1
                if i >= curr_value: 
                    curr_value = i
                    my_newaction = counter

            #Update q value from this experience tuple
            self.q_table[state-1, my_action] = (1-self.alpha)*self.q_table[state-1, my_action] + self.alpha*(reward + self.gamma*self.q_table[new_state-1, my_newaction])
        
        #epsilon greedy with epsilon decay
        if random.random() <= self.epsilon: 
            action = random.randint(0,3)
            self.epsilon = self.epsilon*self.epsilon_decay  


        return action

    def test (self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)

        #select best action
        counter = -1
        action = 0
        curr_value = -10000000
        for i in self.q_table[s-1]:
            counter+=1 
            if i >= curr_value: 
                curr_value = i
                action == counter
            
        #update state action pair
        self.curr_state_action = [s, action]
        # print(self.q_table)
        return action