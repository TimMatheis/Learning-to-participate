"""LIO with policy gradient for policy optimization."""
import numpy as np
import tensorflow as tf

import lio.utils.util as util

import random


class LIO(object):

    def __init__(self, config, l_obs, l_action, nn, agent_name,
                 r_multiplier=2, n_agents=1, agent_id=0):
        self.alg_name = 'lio'
        self.l_obs = l_obs
        self.l_action = l_action

        # ---------------------------------------------------------------
        # self.l_obs_trade = 10
        # self.l_action_trade = 3
        # --------------------------------------------------------------

        self.nn = nn
        self.agent_name = agent_name
        self.r_multiplier = r_multiplier
        self.n_agents = n_agents
        self.agent_id = agent_id

        self.list_other_id = list(range(0, self.n_agents))
        del self.list_other_id[self.agent_id]

        # Default is allow the agent to give rewards
        self.can_give = True

        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_reward = config.lr_reward
        if 'optimizer' in config:
            self.optimizer = config.optimizer
        else:
            self.optimizer = 'sgd'
        self.reg = config.reg
        self.reg_coeff = config.reg_coeff

        self.q_table = np.zeros([l_obs, l_action])
        # self.q_table_trade = np.zeros([l_obs_trade, l_action_trade])

        # self.create_networks()
        # self.policy_new = PolicyNew
        

    def receive_list_of_agents(self, list_of_agents):
        self.list_of_agents = list_of_agents


    def run_trade(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.randint(self.l_action-1-9, self.l_action-1) # Explore action space
            print("RANDOM ac agent", action)
        else:
            state = np.where(state == 1)
            # allowed_actions = [self.l_action-1-10 : self.l_action-1]
            action = np.argmax(self.q_table[state, self.l_action-1-9 : self.l_action])  + 6 # Exploit learned values
            # print(allowed_actions)
            print(self.q_table[state, self.l_action-1-9 : self.l_action])
            print("ac agent", action)   
        return action


    def run_actor(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, self.l_action-1-10) # Explore action space
        else:
            state = np.where(state == 1)
            # allowed_actions = [0, self.l_action-1-10]
            action = np.argmax(self.q_table[state, : self.l_action-1-9]) # Exploit learned values
        return action


    def update(self, state, action, next_state, reward):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # correct this! -> shouldn't be hard-coded
        gamma = 0.99
        alpha = 0.001 # 0.001
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_table[state, action] = new_value
        # print("Update q", self.q_table)

    
    def run_actor_basic(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, self.l_action-1) # Explore action space
            print("RANDOM ac agent", action)
        else:
            state = np.where(state == 1)
            # allowed_actions = [self.l_action-1-10 : self.l_action-1]
            action = np.argmax(self.q_table[state, : self.l_action]) # Exploit learned values
            # print(allowed_actions)
            print(self.q_table)
            print("ac agent", action)   
        return action

